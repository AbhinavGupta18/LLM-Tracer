#include "model_selector.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

using namespace ftxui;

const ModelPreset& runModelSelector() {
    int selected = 0;
    int n = (int)PRESETS.size();

    auto screen = ScreenInteractive::Fullscreen();

    auto renderer = Renderer([&]() -> Element {
        Elements rows;

        for (int i = 0; i < n; i++) {
            auto& p = PRESETS[i];

            std::string layers_s = std::to_string(p.n_layers) + " layers";
            std::string dim_s    = std::to_string(p.hidden_dim) + " dim";

            Element name_col  = text(" " + p.display_name + " ") | size(WIDTH, EQUAL, 22);
            Element stats_col = text(layers_s + "  " + dim_s + "  " + p.dtype)
                                | color(Color::GrayLight)
                                | size(WIDTH, EQUAL, 32);
            Element notes_col = text(p.notes) | color(Color::GrayDark);

            Element row = hbox({name_col, stats_col, notes_col});

            if (i == selected) {
                row = row | inverted;
            }

            rows.push_back(row);
        }

        return vbox({
            text("") | size(HEIGHT, EQUAL, 2),
            text("  Select a model to simulate") | bold,
            text("  j/k or arrows to navigate, Enter to confirm") | color(Color::GrayDark),
            text("") | size(HEIGHT, EQUAL, 1),
            window(text(" Available Models "), vbox(rows)),
            text("") | size(HEIGHT, EQUAL, 1),
            text("  All models run in demo mode -- no GGUF file needed") | color(Color::GrayDark),
        });
    });

    auto root = CatchEvent(renderer, [&](Event e) -> bool {
        if (e == Event::Character('j') || e == Event::ArrowDown) {
            selected = std::min(selected + 1, n - 1);
            return true;
        }
        if (e == Event::Character('k') || e == Event::ArrowUp) {
            selected = std::max(selected - 1, 0);
            return true;
        }
        if (e == Event::Return) {
            screen.ExitLoopClosure()();
            return true;
        }
        if (e == Event::Character('q') || e == Event::Character('Q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    screen.Loop(root);
    return PRESETS[selected];
}
