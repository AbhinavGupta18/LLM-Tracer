#include "tui_app.hpp"

#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <thread>

using namespace ftxui;

// UTF-8 block characters used for visualization
static const char* BLOCK_FULL   = "\xE2\x96\x88";  // U+2588
static const char* BLOCK_DARK   = "\xE2\x96\x93";  // U+2593
static const char* BLOCK_MED    = "\xE2\x96\x92";  // U+2592
static const char* BLOCK_LIGHT  = "\xE2\x96\x91";  // U+2591
static const char* TRI_DOWN     = "\xE2\x96\xBC";  // U+25BC
static const char* TRI_RIGHT    = "\xE2\x96\xB6";  // U+25B6
static const char* CIRCLE       = "\xE2\x97\x8F";  // U+25CF

TUIApp::TUIApp(ModelHook* hook, std::vector<LayerNode> topology)
    : hook_(hook), topology_(std::move(topology)) {}

std::string TUIApp::formatTimestamp(
    std::chrono::time_point<std::chrono::system_clock> tp) const {
    auto t  = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  tp.time_since_epoch()) % 1000;
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                  (int)ms.count());
    return buf;
}

// returns the most recent event for the pinned layer, or the last event overall
const LayerEvent* TUIApp::findPinnedEvent(const std::vector<LayerEvent>& events) const {
    if (events.empty()) return nullptr;
    if (pinned_layer_.empty()) return &events.back();
    for (int i = (int)events.size() - 1; i >= 0; i--) {
        if (events[i].layer_name.find(pinned_layer_) != std::string::npos) {
            return &events[i];
        }
    }
    return &events.back();
}

// ---- panel 1: model topology --------------------------------------------

Element TUIApp::renderTopology(bool focused) {
    Elements rows;
    int visit = 0;
    visible_node_names_.clear();

    std::function<void(const LayerNode&, int)> walk;
    walk = [&](const LayerNode& node, int depth) {
        bool selected  = (visit == topo_cursor_);
        bool is_pinned = pinned_layer_.empty()
                             ? node.active_capture
                             : (node.name == pinned_layer_);
        visible_node_names_.push_back(node.name);
        visit++;

        std::string pad(depth * 2, ' ');
        std::string marker;
        if (node.children.empty()) {
            marker = is_pinned ? std::string(" ") + CIRCLE + " " : "   ";
        } else {
            marker = std::string(" ") +
                     (node.expanded ? TRI_DOWN : TRI_RIGHT) + " ";
        }

        std::string label = pad + marker + node.name;
        if (!node.children.empty() && is_pinned) {
            label += "  [Active Capture Target]";
        }

        Element line = text(label);
        if (selected) {
            line = line | inverted;
        } else if (is_pinned) {
            line = line | color(Color::Yellow);
        }
        rows.push_back(line);

        if (node.expanded) {
            for (auto& child : node.children) walk(child, depth + 1);
        }
    };

    for (auto& root : topology_) walk(root, 0);

    topo_cursor_ = std::clamp(topo_cursor_, 0, std::max(0, visit - 1));

    std::string title = focused
        ? std::string(" ") + BLOCK_FULL + " 1. MODEL TOPOLOGY (Focus Active) "
        : " 1. MODEL TOPOLOGY ";

    return window(
        text(title) | (focused ? bold : color(Color::GrayLight)),
        vbox(rows) | vscroll_indicator | frame
    );
}

// ---- panel 2: live packet stream ----------------------------------------

Element TUIApp::renderPacketStream(bool focused) {
    std::string title = focused
        ? " 2. LIVE PACKET STREAM (Focus) "
        : " 2. LIVE PACKET STREAM ";

    auto events = hook_->events.snapshot();

    Elements rows;
    rows.push_back(
        hbox({
            text("  ID  ") | bold,
            text(" | ") | color(Color::GrayDark),
            text(" TIMESTAMP    ") | bold,
            text(" | ") | color(Color::GrayDark),
            text(" LAYER TYPE   ") | bold,
            text(" | ") | color(Color::GrayDark),
            text(" COMPUTE DEVICE   ") | bold,
        }) | color(Color::GrayLight)
    );
    rows.push_back(separator());

    if (events.empty()) {
        rows.push_back(text("  Waiting for events...") | color(Color::GrayDark));
    } else {
        int start = std::max(0, (int)events.size() - 14);
        for (int i = start; i < (int)events.size(); i++) {
            auto& e = events[i];

            std::string id_s = std::to_string(e.id);
            while ((int)id_s.size() < 3) id_s = " " + id_s;

            std::string ty_s = layerTypeStr(e.type);
            while ((int)ty_s.size() < 12) ty_s += " ";

            bool cpu = e.device.find("CPU") != std::string::npos;
            Color row_col = cpu ? Color::Yellow : Color::White;

            rows.push_back(
                hbox({
                    text("  " + id_s + "  "),
                    text(" | ") | color(Color::GrayDark),
                    text(" " + formatTimestamp(e.timestamp) + " "),
                    text(" | ") | color(Color::GrayDark),
                    text(" " + ty_s + " "),
                    text(" | ") | color(Color::GrayDark),
                    text(" " + e.device + " "),
                }) | color(row_col)
            );
        }
    }

    return window(
        text(title) | (focused ? bold : color(Color::GrayLight)),
        vbox(rows) | vscroll_indicator | frame
    );
}

// ---- panel 3: attention matrix ------------------------------------------

Element TUIApp::renderAttentionMatrix(bool focused) {
    std::string title = focused
        ? " 3. ATTENTION MATRIX VISUALIZER (HEAD 0) (Focus) "
        : " 3. ATTENTION MATRIX VISUALIZER (HEAD 0) ";

    auto events = hook_->events.snapshot();

    // prefer the pinned layer's attention data, fall back to latest attention event
    const LayerEvent* ev = nullptr;
    if (!pinned_layer_.empty()) {
        for (int i = (int)events.size() - 1; i >= 0; i--) {
            if (events[i].type == LayerType::Attention &&
                !events[i].attn_weights.empty() &&
                events[i].layer_name.find(pinned_layer_) != std::string::npos) {
                ev = &events[i];
                break;
            }
        }
    }
    if (!ev) {
        for (int i = (int)events.size() - 1; i >= 0; i--) {
            if (events[i].type == LayerType::Attention &&
                !events[i].attn_weights.empty()) {
                ev = &events[i];
                break;
            }
        }
    }

    if (!ev) {
        return window(
            text(title) | (focused ? bold : color(Color::GrayLight)),
            text("  Waiting for attention layer data...") | color(Color::GrayDark)
        );
    }

    int seq = (int)ev->attn_weights.size();
    int vx  = std::clamp(attn_pan_x_, 0, std::max(0, seq - 1));
    int vy  = std::clamp(attn_pan_y_, 0, std::max(0, seq - 1));
    int vw  = std::min(8, seq - vx);
    int vh  = std::min(6, seq - vy);

    auto cellColor = [&](float w) -> std::pair<std::string, Color> {
        float adj = std::clamp(w * attn_contrast_, 0.0f, 1.0f);
        if (adj > 0.75f) return {std::string(BLOCK_FULL)  + BLOCK_FULL,  Color::White};
        if (adj > 0.50f) return {std::string(BLOCK_DARK)  + BLOCK_DARK,  Color::GrayLight};
        if (adj > 0.25f) return {std::string(BLOCK_MED)   + BLOCK_MED,   Color::GrayDark};
        if (adj > 0.05f) return {std::string(BLOCK_LIGHT) + BLOCK_LIGHT, Color::GrayDark};
        return {"  ", Color::GrayDark};
    };

    // header token labels
    Elements hdr_cells;
    hdr_cells.push_back(text(std::string(11, ' ')));
    for (int c = vx; c < vx + vw; c++) {
        std::string lbl = (c < (int)ev->token_labels.size())
            ? ev->token_labels[c] : "?";
        if ((int)lbl.size() > 6) lbl = lbl.substr(0, 6);
        std::string cell = "[" + lbl + "]";
        cell += std::string(std::max(0, 9 - (int)cell.size()), ' ');
        hdr_cells.push_back(text(cell));
    }

    Elements matrix_rows;
    matrix_rows.push_back(hbox(hdr_cells));

    for (int r = vy; r < vy + vh; r++) {
        Elements row_cells;

        std::string lbl = (r < (int)ev->token_labels.size())
            ? ev->token_labels[r] : "?";
        if ((int)lbl.size() > 6) lbl = lbl.substr(0, 6);
        std::string prefix = "[" + lbl + "]";
        prefix += std::string(std::max(0, 11 - (int)prefix.size()), ' ');
        row_cells.push_back(text(prefix));

        for (int c = vx; c < vx + vw; c++) {
            float w = (c < (int)ev->attn_weights[r].size())
                ? ev->attn_weights[r][c] : 0.0f;
            auto [block, col] = cellColor(w);
            row_cells.push_back(text(block) | color(col));
            row_cells.push_back(text("       "));
        }
        matrix_rows.push_back(hbox(row_cells));
    }

    std::string viewport = "Viewport Window: [" + std::to_string(vx) + "-" +
                           std::to_string(vx + vw - 1) + "] x [" +
                           std::to_string(vy) + "-" +
                           std::to_string(vy + vh - 1) + "]";

    Element help = vbox({
        text("[Focus + F]: Open Fullscreen") | color(Color::GrayDark),
        text("[Arrows/(h,j,k,l)]: Pan Matrix") | color(Color::GrayDark),
        text("[+/-]: Change Weight Contrast") | color(Color::GrayDark),
    }) | size(WIDTH, EQUAL, 38);

    return window(
        text(title) | (focused ? bold : color(Color::GrayLight)),
        vbox({
            hbox({vbox(matrix_rows) | flex, separator(), help}),
            text("  " + viewport) | color(Color::GrayDark),
        })
    );
}

// ---- panel 4: runtime metrics -------------------------------------------

Element TUIApp::renderMetrics(bool focused) {
    std::string title = focused
        ? " 4. RUNTIME METRICS INSPECTOR (Focus) "
        : " 4. RUNTIME METRICS INSPECTOR ";

    auto events = hook_->events.snapshot();
    const LayerEvent* evp = findPinnedEvent(events);
    if (!evp) {
        return window(
            text(title) | (focused ? bold : color(Color::GrayLight)),
            text("  No data yet") | color(Color::GrayDark)
        );
    }

    auto& ev = *evp;
    auto& t  = ev.tensor;

    // shape string
    std::string shape_s;
    for (size_t i = 0; i < t.shape.size(); i++) {
        if (i) shape_s += ", ";
        shape_s += std::to_string(t.shape[i]);
    }
    shape_s = "[" + shape_s + "]";

    // sparsity bar using colored block chars
    static const int BAR_W = 13;
    int filled = (int)(t.sparsity * BAR_W);
    Elements bar;
    for (int i = 0; i < BAR_W; i++) {
        if (i < filled) {
            bar.push_back(text(BLOCK_FULL) | color(Color::Green));
        } else if (i == filled && filled < BAR_W) {
            bar.push_back(text(BLOCK_FULL) | color(Color::Yellow));
        } else {
            bar.push_back(text(BLOCK_LIGHT) | color(Color::GrayDark));
        }
    }

    std::ostringstream pct;
    pct << std::fixed << std::setprecision(1) << (t.sparsity * 100.0f) << "%";

    // latency
    bool normal = ev.latency_ms < 2.5;
    std::ostringstream lat;
    lat << std::fixed << std::setprecision(3) << ev.latency_ms << " ms";
    std::string lat_s = lat.str() +
                        (normal ? "  (Within Normal Bounds)" : "  (High Latency)");

    return window(
        text(title) | (focused ? bold : color(Color::GrayLight)),
        vbox({
            hbox({
                text(" Tensor Shape : "),
                text(shape_s) | bold,
                text("   Dtype: "),
                text(t.dtype) | bold,
            }),
            hbox({
                text(" Sparsity Rate: "),
                hbox(bar),
                text(" " + pct.str()) | bold,
            }),
            hbox({
                text(" Latency Delta: "),
                text(lat_s) | color(normal ? Color::Green : Color::Yellow),
            }),
        })
    );
}

// ---- panel 5: anomaly ledger --------------------------------------------

Element TUIApp::renderAnomalyLedger(bool focused) {
    std::string title = focused
        ? " 5. NUMERICAL ANOMALY LEDGER (Focus) "
        : " 5. NUMERICAL ANOMALY LEDGER ";

    auto events = hook_->events.snapshot();

    Elements rows;
    for (auto& ev : events) {
        if (!ev.has_anomaly) continue;

        bool oom = ev.anomaly_desc.find("OOM") != std::string::npos ||
                   ev.anomaly_desc.find("CPU") != std::string::npos;
        Color c   = oom ? Color::Red : Color::Yellow;
        std::string icon = oom ? " x " : " ! ";

        rows.push_back(
            hbox({
                text(" " + formatTimestamp(ev.timestamp)) | color(Color::GrayLight),
                text(icon) | color(c) | bold,
                text(ev.anomaly_desc) | color(c),
            })
        );
    }

    if (rows.empty()) {
        rows.push_back(text("  No anomalies detected.") | color(Color::Green));
    }

    return window(
        text(title) | (focused ? bold : color(Color::GrayLight)),
        vbox(rows) | vscroll_indicator | frame
    );
}

// ---- status bar ---------------------------------------------------------

Element TUIApp::renderStatusBar() {
    static const char* labels[] = {
        "1:Topology", "2:Stream", "3:Attention", "4:Metrics", "5:Anomaly"
    };

    Elements tabs;
    for (int i = 0; i < 5; i++) {
        Element t = text(std::string(" ") + labels[i] + " ");
        tabs.push_back(i == focused_panel_ ? (t | inverted) : t);
        if (i < 4) tabs.push_back(text(" | ") | color(Color::GrayDark));
    }

    return hbox({
        text(" [Tab]: Cycle Focus  [Q]: Quit  [j/k]: Navigate  ") |
            color(Color::GrayLight),
        text(" | ") | color(Color::GrayDark),
        hbox(tabs),
    });
}

// ---- layout -------------------------------------------------------------

Element TUIApp::renderLayout() {
    if (attn_fullscreen_) {
        return vbox({
            renderAttentionMatrix(true) | flex,
            renderStatusBar(),
        });
    }

    bool f0 = focused_panel_ == 0;
    bool f1 = focused_panel_ == 1;
    bool f2 = focused_panel_ == 2;
    bool f3 = focused_panel_ == 3;
    bool f4 = focused_panel_ == 4;

    return vbox({
        hbox({
            renderTopology(f0)      | size(WIDTH, EQUAL, 44),
            renderPacketStream(f1)  | flex,
        }) | size(HEIGHT, EQUAL, 17),
        renderAttentionMatrix(f2) | size(HEIGHT, EQUAL, 11),
        hbox({
            renderMetrics(f3)       | flex,
            renderAnomalyLedger(f4) | flex,
        }) | size(HEIGHT, EQUAL, 7),
        renderStatusBar(),
    });
}

// ---- event handling -----------------------------------------------------

bool TUIApp::handleEvent(Event e) {
    // F exits fullscreen from anywhere, not just when panel 2 is focused
    if (attn_fullscreen_ && (e == Event::Character('f') || e == Event::Character('F'))) {
        attn_fullscreen_ = false;
        return true;
    }

    if (e == Event::Tab) {
        focused_panel_ = (focused_panel_ + 1) % 5;
        return true;
    }

    switch (focused_panel_) {
    case 0: // topology navigation
        if (e == Event::Character('j') || e == Event::ArrowDown) {
            topo_cursor_++;
            return true;
        }
        if (e == Event::Character('k') || e == Event::ArrowUp) {
            topo_cursor_ = std::max(0, topo_cursor_ - 1);
            return true;
        }
        if (e == Event::Character(' ')) {
            if (topo_cursor_ < (int)visible_node_names_.size()) {
                pinned_layer_ = visible_node_names_[topo_cursor_];
            }
            return true;
        }
        break;

    case 2: // attention matrix pan / zoom
        if (e == Event::Character('h') || e == Event::ArrowLeft) {
            attn_pan_x_ = std::max(0, attn_pan_x_ - 1);
            return true;
        }
        if (e == Event::Character('l') || e == Event::ArrowRight) {
            attn_pan_x_++;
            return true;
        }
        if (e == Event::Character('k') || e == Event::ArrowUp) {
            attn_pan_y_ = std::max(0, attn_pan_y_ - 1);
            return true;
        }
        if (e == Event::Character('j') || e == Event::ArrowDown) {
            attn_pan_y_++;
            return true;
        }
        if (e == Event::Character('+') || e == Event::Character('=')) {
            attn_contrast_ = std::min(4.0f, attn_contrast_ + 0.25f);
            return true;
        }
        if (e == Event::Character('-')) {
            attn_contrast_ = std::max(0.25f, attn_contrast_ - 0.25f);
            return true;
        }
        if (e == Event::Character('f') || e == Event::Character('F')) {
            attn_fullscreen_ = !attn_fullscreen_;
            return true;
        }
        break;

    default:
        break;
    }

    return false;
}

// ---- main loop ----------------------------------------------------------

void TUIApp::run() {
    auto screen = ScreenInteractive::Fullscreen();

    auto renderer = Renderer([&]() -> Element {
        return renderLayout();
    });

    auto root = CatchEvent(renderer, [&](Event e) -> bool {
        if (e == Event::Character('q') || e == Event::Character('Q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        return handleEvent(e);
    });

    std::atomic<bool> alive{true};
    std::thread refresh([&] {
        while (alive.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(root);

    alive.store(false);
    refresh.join();
}
