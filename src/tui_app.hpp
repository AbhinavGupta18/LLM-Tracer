#pragma once

#include "model_hook.hpp"
#include "layer_event.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <vector>
#include <string>
#include <chrono>

class TUIApp {
public:
    TUIApp(ModelHook* hook, std::vector<LayerNode> topology);
    void run();

private:
    ModelHook*             hook_;
    std::vector<LayerNode> topology_;

    int   focused_panel_    = 0;
    int   topo_cursor_      = 0;
    int   attn_pan_x_       = 0;
    int   attn_pan_y_       = 0;
    float attn_contrast_    = 1.0f;
    bool  attn_fullscreen_  = false;

    std::string              pinned_layer_;        // layer name selected with Space
    std::vector<std::string> visible_node_names_;  // rebuilt each render for Space lookup

    const LayerEvent* findPinnedEvent(const std::vector<LayerEvent>& events) const;

    ftxui::Element renderTopology(bool focused);
    ftxui::Element renderPacketStream(bool focused);
    ftxui::Element renderAttentionMatrix(bool focused);
    ftxui::Element renderMetrics(bool focused);
    ftxui::Element renderAnomalyLedger(bool focused);
    ftxui::Element renderStatusBar();
    ftxui::Element renderLayout();

    bool handleEvent(ftxui::Event e);

    std::string formatTimestamp(
        std::chrono::time_point<std::chrono::system_clock> tp) const;
};
