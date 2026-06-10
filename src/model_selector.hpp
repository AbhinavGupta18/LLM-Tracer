#pragma once

#include "model_preset.hpp"

// Shows a fullscreen model picker and returns the chosen preset.
// Blocks until the user confirms with Enter.
const ModelPreset& runModelSelector();
