// rexgpu-metal-runner: runs a list of small Metal jobs described in JSON and
// dumps raw outputs. It knows nothing about Xenos; the Python driver
// (tools/gputest/shaderdiff.py) builds the jobs for each translation ABI.
//
// usage: rexgpu-metal-runner <jobs.json>
//
// jobs.json is an array of jobs:
// {
//   "mode": "render" | "compute",
//   "out": "path/out.bin",
//   // compute: one source, kernel function, thread count; writes the named
//   // buffer(s) listed in "dump" to "out" (concatenated).
//   "source": "path.metal", "function": "name", "threads": N, "dump": ["b"],
//   // render: vertex + fragment (may be the same source), triangle list.
//   "vertex_source": "...", "vertex_function": "...",
//   "fragment_source": "...", "fragment_function": "...",
//   "vertex_count": N, "width": W, "height": H, "dump": ["buf", ...] (also render),
//   "colors": BITMASK (RGBA32Float targets), "depth": bool,
//   // writes each enabled color target (W*H*16 bytes) then depth (W*H*4).
//   "buffers": [{"name": "n", "file": "f.bin" | "size": BYTES,
//                "bind": [["vertex"|"fragment"|"compute", INDEX], ...],
//                "patch": [[OFFSET, "buffer"|"texture"|"sampler", "name"], ...]}],
//   "textures": [{"name": "t", "type": "2d_array"|"3d"|"cube", "format":
//                 "rgba16f"|"rgba8snorm"|"rgba32f", "size": [W,H,D], "mips": M,
//                 "fill": [r,g,b,a], "bind": [["fragment", INDEX], ...]}],
//   "samplers": [{"name": "s", "linear": bool, "bind": [["fragment", INDEX]]}]
// }
// Buffers are created first (contents from file or zeroed), then textures and
// samplers, then patches (64-bit GPU addresses / resource IDs) are applied.
// Prints one line per job: "JOB <i> ok" or "JOB <i> error <message>".

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

id<MTLDevice> g_dev;
id<MTLCommandQueue> g_queue;
std::map<std::string, id<MTLLibrary>> g_library_cache;

std::string Str(id o) { return o ? std::string([[o description] UTF8String]) : std::string(); }

id<MTLLibrary> LoadLibrary(NSString* path, std::string& err) {
  std::string key = path.UTF8String;
  auto it = g_library_cache.find(key);
  if (it != g_library_cache.end()) {
    return it->second;
  }
  NSError* e = nil;
  id<MTLLibrary> lib = nil;
  if ([path hasSuffix:@".metallib"]) {
    lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
  } else {
    NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&e];
    if (src) {
      MTLCompileOptions* opts = [MTLCompileOptions new];
      opts.mathMode = MTLMathModeSafe;
      opts.languageVersion = MTLLanguageVersion3_1;
      lib = [g_dev newLibraryWithSource:src options:opts error:&e];
    }
  }
  if (!lib) {
    err = "compile: " + Str(e.localizedDescription);
    std::replace(err.begin(), err.end(), '\n', ' ');
    return nil;
  }
  if (g_library_cache.size() > 64) {
    g_library_cache.clear();
  }
  g_library_cache[key] = lib;
  return lib;
}

MTLPixelFormat Format(NSString* f) {
  if ([f isEqualToString:@"rgba8snorm"]) return MTLPixelFormatRGBA8Snorm;
  if ([f isEqualToString:@"rgba32f"]) return MTLPixelFormatRGBA32Float;
  if ([f isEqualToString:@"rgba8unorm"]) return MTLPixelFormatRGBA8Unorm;
  return MTLPixelFormatRGBA16Float;
}

struct Resources {
  std::map<std::string, id<MTLBuffer>> buffers;
  std::map<std::string, id<MTLTexture>> textures;
  std::map<std::string, id<MTLSamplerState>> samplers;
  std::vector<id<MTLResource>> all;
};

bool CreateResources(NSDictionary* job, Resources& r, std::string& err) {
  for (NSDictionary* b in job[@"buffers"]) {
    std::string name = [b[@"name"] UTF8String];
    NSData* data = nil;
    size_t size = [b[@"size"] unsignedLongValue];
    if (b[@"file"]) {
      data = [NSData dataWithContentsOfFile:b[@"file"]];
      if (!data) {
        err = "missing buffer file " + std::string([b[@"file"] UTF8String]);
        return false;
      }
      size = std::max<size_t>(size, data.length);
    }
    size = std::max<size_t>(size, 16);
    id<MTLBuffer> buf = [g_dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    std::memset(buf.contents, 0, size);
    if (data) {
      std::memcpy(buf.contents, data.bytes, data.length);
    }
    r.buffers[name] = buf;
    r.all.push_back(buf);
  }
  for (NSDictionary* t in job[@"textures"]) {
    NSString* type = t[@"type"];
    NSArray* sz = t[@"size"];
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.pixelFormat = Format(t[@"format"]);
    td.width = [sz[0] unsignedIntValue];
    td.height = [sz[1] unsignedIntValue];
    uint32_t depth = sz.count > 2 ? [sz[2] unsignedIntValue] : 1;
    if ([type isEqualToString:@"3d"]) {
      td.textureType = MTLTextureType3D;
      td.depth = depth;
    } else if ([type isEqualToString:@"cube"]) {
      td.textureType = MTLTextureTypeCube;
    } else if ([type isEqualToString:@"2d"]) {
      td.textureType = MTLTextureType2D;
    } else {
      td.textureType = MTLTextureType2DArray;
      td.arrayLength = depth;
    }
    td.mipmapLevelCount = t[@"mips"] ? [t[@"mips"] unsignedIntValue] : 1;
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> tex = [g_dev newTextureWithDescriptor:td];
    NSArray* fill = t[@"fill"];
    float f[4] = {0, 0, 0, 0};
    for (NSUInteger i = 0; i < 4 && i < fill.count; ++i) f[i] = [fill[i] floatValue];
    uint32_t slices = td.textureType == MTLTextureTypeCube ? 6 : (td.textureType == MTLTextureType2DArray ? depth : 1);
    for (uint32_t m = 0; m < td.mipmapLevelCount; ++m) {
      uint32_t w = std::max<uint32_t>(1, uint32_t(td.width) >> m);
      uint32_t h = std::max<uint32_t>(1, uint32_t(td.height) >> m);
      uint32_t d = td.textureType == MTLTextureType3D ? std::max<uint32_t>(1, depth >> m) : 1;
      // "pattern": "gradient" makes every texel, mip and slice distinct so a
      // wrong coordinate, LOD, face or texture binding changes the result.
      bool gradient = [t[@"pattern"] isEqualToString:@"gradient"];
      size_t bpp = td.pixelFormat == MTLPixelFormatRGBA32Float   ? 16
                   : td.pixelFormat == MTLPixelFormatRGBA16Float ? 8
                                                                 : 4;
      std::vector<uint8_t> all_slices;
      for (uint32_t s = 0; s < slices; ++s) {
        std::vector<uint8_t> px(size_t(w) * h * d * bpp);
        for (uint32_t z = 0; z < d; ++z)
          for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
              float v[4] = {f[0], f[1], f[2], f[3]};
              if (gradient) {
                v[0] = f[0] * ((x + 0.5f) / w);
                v[1] = f[1] * ((y + 0.5f) / h);
                v[2] = f[2] * (1.0f - 0.11f * m - 0.07f * s - 0.05f * (z * 4.0f / std::max<uint32_t>(1, d)));
                v[3] = f[3] * (((x ^ y ^ z) & 1) ? 1.0f : 0.25f);
              }
              size_t i = (size_t(z) * h + y) * w + x;
              if (bpp == 16) {
                std::memcpy(&px[i * 16], v, 16);
              } else if (bpp == 8) {
                __fp16 hf[4] = {__fp16(v[0]), __fp16(v[1]), __fp16(v[2]), __fp16(v[3])};
                std::memcpy(&px[i * 8], hf, 8);
              } else {
                for (int c = 0; c < 4; ++c) {
                  float q = td.pixelFormat == MTLPixelFormatRGBA8Snorm
                                ? std::clamp(v[c], -1.0f, 1.0f) * 127.0f
                                : std::clamp(v[c], 0.0f, 1.0f) * 255.0f;
                  int qi = int(q + (q >= 0 ? 0.5f : -0.5f));
                  px[i * 4 + c] = td.pixelFormat == MTLPixelFormatRGBA8Snorm ? uint8_t(int8_t(qi)) : uint8_t(qi);
                }
              }
            }
        [tex replaceRegion:MTLRegionMake3D(0, 0, 0, w, h, d)
               mipmapLevel:m
                     slice:s
                 withBytes:px.data()
               bytesPerRow:w * bpp
             bytesPerImage:size_t(w) * h * bpp];
      }
    }
    r.textures[[t[@"name"] UTF8String]] = tex;
    r.all.push_back(tex);
  }
  for (NSDictionary* s in job[@"samplers"]) {
    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.supportArgumentBuffers = YES;
    bool linear = [s[@"linear"] boolValue];
    sd.minFilter = sd.magFilter = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.mipFilter = linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = MTLSamplerAddressModeRepeat;
    r.samplers[[s[@"name"] UTF8String]] = [g_dev newSamplerStateWithDescriptor:sd];
  }
  for (NSDictionary* b in job[@"buffers"]) {
    id<MTLBuffer> buf = r.buffers[[b[@"name"] UTF8String]];
    for (NSArray* p in b[@"patch"]) {
      size_t off = [p[0] unsignedLongValue];
      NSString* kind = p[1];
      std::string name = [p[2] UTF8String];
      uint64_t v = 0;
      if ([kind isEqualToString:@"buffer"]) {
        if (!r.buffers.count(name)) { err = "patch: no buffer " + name; return false; }
        v = r.buffers[name].gpuAddress;
        if (p.count > 3) v += [p[3] unsignedLongLongValue];
      } else if ([kind isEqualToString:@"texture"]) {
        if (!r.textures.count(name)) { err = "patch: no texture " + name; return false; }
        MTLResourceID id = r.textures[name].gpuResourceID;
        std::memcpy(&v, &id, 8);
      } else {
        if (!r.samplers.count(name)) { err = "patch: no sampler " + name; return false; }
        MTLResourceID id = r.samplers[name].gpuResourceID;
        std::memcpy(&v, &id, 8);
      }
      if (off + 8 > buf.length) { err = "patch out of range"; return false; }
      std::memcpy((uint8_t*)buf.contents + off, &v, 8);
    }
  }
  return true;
}

template <typename Enc>
void BindAll(NSDictionary* job, Resources& r, Enc enc, NSString* stage) {
  auto matches = [&](NSArray* bind, void (^fn)(NSUInteger)) {
    for (NSArray* bi in bind) {
      if ([bi[0] isEqualToString:stage]) fn([bi[1] unsignedIntegerValue]);
    }
  };
  bool render = [enc conformsToProtocol:@protocol(MTLRenderCommandEncoder)];
  bool vertex = [stage isEqualToString:@"vertex"];
  for (NSDictionary* b in job[@"buffers"]) {
    id<MTLBuffer> buf = r.buffers[[b[@"name"] UTF8String]];
    matches(b[@"bind"], ^(NSUInteger i) {
      if (!render) [(id<MTLComputeCommandEncoder>)enc setBuffer:buf offset:0 atIndex:i];
      else if (vertex) [(id<MTLRenderCommandEncoder>)enc setVertexBuffer:buf offset:0 atIndex:i];
      else [(id<MTLRenderCommandEncoder>)enc setFragmentBuffer:buf offset:0 atIndex:i];
    });
  }
  for (NSDictionary* t in job[@"textures"]) {
    id<MTLTexture> tex = r.textures[[t[@"name"] UTF8String]];
    matches(t[@"bind"], ^(NSUInteger i) {
      if (!render) [(id<MTLComputeCommandEncoder>)enc setTexture:tex atIndex:i];
      else if (vertex) [(id<MTLRenderCommandEncoder>)enc setVertexTexture:tex atIndex:i];
      else [(id<MTLRenderCommandEncoder>)enc setFragmentTexture:tex atIndex:i];
    });
  }
  for (NSDictionary* s in job[@"samplers"]) {
    id<MTLSamplerState> smp = r.samplers[[s[@"name"] UTF8String]];
    matches(s[@"bind"], ^(NSUInteger i) {
      if (!render) [(id<MTLComputeCommandEncoder>)enc setSamplerState:smp atIndex:i];
      else if (vertex) [(id<MTLRenderCommandEncoder>)enc setVertexSamplerState:smp atIndex:i];
      else [(id<MTLRenderCommandEncoder>)enc setFragmentSamplerState:smp atIndex:i];
    });
  }
}

bool RunJob(NSDictionary* job, std::string& err) {
  Resources r;
  if (!CreateResources(job, r, err)) return false;
  NSMutableData* out = [NSMutableData data];
  id<MTLCommandBuffer> cb = [g_queue commandBuffer];
  if ([job[@"mode"] isEqualToString:@"compute"]) {
    id<MTLLibrary> lib = LoadLibrary(job[@"source"], err);
    if (!lib) return false;
    id<MTLFunction> fn = [lib newFunctionWithName:job[@"function"]];
    if (!fn) { err = "no function"; return false; }
    NSError* e = nil;
    id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithFunction:fn error:&e];
    if (!ps) { err = "pipeline: " + Str(e.localizedDescription); return false; }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    BindAll(job, r, enc, @"compute");
    for (auto res : r.all) [enc useResource:res usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    NSUInteger n = [job[@"threads"] unsignedIntegerValue];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(n, 64), 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { err = "gpu: " + Str(cb.error.localizedDescription); return false; }
    for (NSString* name in job[@"dump"]) {
      id<MTLBuffer> b = r.buffers[name.UTF8String];
      [out appendBytes:b.contents length:b.length];
    }
  } else {
    id<MTLLibrary> vlib = LoadLibrary(job[@"vertex_source"], err);
    if (!vlib) return false;
    id<MTLLibrary> flib = LoadLibrary(job[@"fragment_source"], err);
    if (!flib) return false;
    uint32_t W = [job[@"width"] unsignedIntValue], H = [job[@"height"] unsignedIntValue];
    uint32_t colors = [job[@"colors"] unsignedIntValue];
    bool depth = [job[@"depth"] boolValue];
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [vlib newFunctionWithName:job[@"vertex_function"]];
    pd.fragmentFunction = [flib newFunctionWithName:job[@"fragment_function"]];
    if (!pd.vertexFunction || !pd.fragmentFunction) { err = "no function"; return false; }
    for (uint32_t i = 0; i < 8; ++i) {
      if (colors & (1u << i)) pd.colorAttachments[i].pixelFormat = MTLPixelFormatRGBA32Float;
    }
    if (depth) pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    NSError* e = nil;
    id<MTLRenderPipelineState> ps = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!ps) {
      err = "pipeline: " + Str(e.localizedDescription);
      std::replace(err.begin(), err.end(), '\n', ' ');
      return false;
    }
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    std::vector<id<MTLTexture>> rts(8, nil);
    for (uint32_t i = 0; i < 8; ++i) {
      if (!(colors & (1u << i))) continue;
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                   width:W height:H mipmapped:NO];
      td.usage = MTLTextureUsageRenderTarget;
      td.storageMode = MTLStorageModeShared;
      rts[i] = [g_dev newTextureWithDescriptor:td];
      rp.colorAttachments[i].texture = rts[i];
      rp.colorAttachments[i].loadAction = MTLLoadActionClear;
      rp.colorAttachments[i].clearColor = MTLClearColorMake(-7, -7, -7, -7);
      rp.colorAttachments[i].storeAction = MTLStoreActionStore;
    }
    id<MTLTexture> dt = nil;
    id<MTLBuffer> dread = nil;
    id<MTLDepthStencilState> ds = nil;
    if (depth) {
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                   width:W height:H mipmapped:NO];
      td.usage = MTLTextureUsageRenderTarget;
      td.storageMode = MTLStorageModePrivate;
      dt = [g_dev newTextureWithDescriptor:td];
      rp.depthAttachment.texture = dt;
      rp.depthAttachment.loadAction = MTLLoadActionClear;
      rp.depthAttachment.clearDepth = 1.0;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
      dd.depthCompareFunction = MTLCompareFunctionAlways;
      dd.depthWriteEnabled = YES;
      ds = [g_dev newDepthStencilStateWithDescriptor:dd];
    }
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:ps];
    if (ds) [enc setDepthStencilState:ds];
    BindAll(job, r, enc, @"vertex");
    BindAll(job, r, enc, @"fragment");
    for (auto res : r.all) {
      [enc useResource:res usage:MTLResourceUsageRead | MTLResourceUsageWrite
                    stages:MTLRenderStageVertex | MTLRenderStageFragment];
    }
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:[job[@"vertex_count"] unsignedIntegerValue]];
    [enc endEncoding];
    if (dt) {
      dread = [g_dev newBufferWithLength:size_t(W) * H * 4 options:MTLResourceStorageModeShared];
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b copyFromTexture:dt sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(W, H, 1) toBuffer:dread destinationOffset:0
     destinationBytesPerRow:W * 4 destinationBytesPerImage:size_t(W) * H * 4];
      [b endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { err = "gpu: " + Str(cb.error.localizedDescription); return false; }
    std::vector<float> px(size_t(W) * H * 4);
    for (uint32_t i = 0; i < 8; ++i) {
      if (!rts[i]) continue;
      [rts[i] getBytes:px.data() bytesPerRow:W * 16 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
      [out appendBytes:px.data() length:px.size() * 4];
    }
    if (dread) [out appendBytes:dread.contents length:dread.length];
    for (NSString* name in job[@"dump"]) {
      id<MTLBuffer> b = r.buffers[name.UTF8String];
      [out appendBytes:b.contents length:b.length];
    }
  }
  if (![out writeToFile:job[@"out"] atomically:NO]) { err = "cannot write out"; return false; }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <jobs.json>\n", argv[0]);
      return 1;
    }
    NSData* d = [NSData dataWithContentsOfFile:@(argv[1])];
    NSError* e = nil;
    NSArray* jobs = d ? [NSJSONSerialization JSONObjectWithData:d options:0 error:&e] : nil;
    if (![jobs isKindOfClass:[NSArray class]]) {
      std::fprintf(stderr, "bad jobs file\n");
      return 1;
    }
    g_dev = MTLCreateSystemDefaultDevice();
    g_queue = [g_dev newCommandQueue];
    int failures = 0;
    for (NSUInteger i = 0; i < jobs.count; ++i) {
      @autoreleasepool {
        std::string err;
        bool ok = RunJob(jobs[i], err);
        if (ok) {
          std::printf("JOB %lu ok\n", (unsigned long)i);
        } else {
          std::printf("JOB %lu error %s\n", (unsigned long)i, err.c_str());
          ++failures;
        }
        std::fflush(stdout);
      }
    }
    return failures ? 3 : 0;
  }
}
