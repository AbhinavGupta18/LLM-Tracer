#include "model_hook.hpp"
#include "model_selector.hpp"
#include "demo_runner.hpp"
#include "tui_app.hpp"

int main() {
    const ModelPreset& preset = runModelSelector();

    ModelHook hook;
    DemoRunner runner(&hook, preset);
    auto topology = runner.buildTopology();
    runner.start();

    TUIApp app(&hook, std::move(topology));
    app.run();

    runner.stop();
    return 0;
}
