/**
 * @file        kernel/xam/text_input.h
 * @brief       A line of text from the player, for the app's own menus
 *
 * The game's keyboard (XamShowKeyboardUI) is for the game; this is the same kind of prompt for the app: a text box
 * over the game (ImGui), the same dialog XamShowKeyboardUI shows. While it is open the game gets no controller
 * input, and the button that confirms it stays masked until released, as with the game's own dialogs.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>

namespace rex::kernel::xam {

// Opens the text input without blocking the caller. done runs on a worker thread with the text
// (UTF-8, at most max_length characters), or nothing when the player cancels or no text box can
// open. false, without calling done, while another text input is open.
bool ShowTextInput(std::string title, std::string default_text, size_t max_length,
                   std::function<void(std::optional<std::string>)> done);

}  // namespace rex::kernel::xam
