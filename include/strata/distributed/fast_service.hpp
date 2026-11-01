#pragma once
#include "strata/distributed/fast_model.hpp"
#include "strata/distributed/fast_execution.hpp"
#include "strata/distributed/multislot_service.hpp"
#include "strata/distributed/session.hpp"

namespace strata::distributed {
// The ordinary JSONL protocol and server adapter remain the public boundary.
// This adapter changes execution to real prompt chunks and slot-row batches.
Failure run_fast_service(MultiSlotConfig, Session&, FastModel&, FastTensorParallel&,
                        const MultiSlotServiceCallbacks&, void* stream, uint32_t timeout_ms);
}
