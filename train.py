# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train"""
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain

from tractorun.run import prepare_and_get_toolbox
from tractorun.backend.tractorch import Tractorch

from urllib.parse import urlparse

import deepspeed

import sys
import socket
import os

def main(input_args=None, overwrite_values=None):
    toolbox = prepare_and_get_toolbox(backend=Tractorch())
    coordinator = toolbox.coordinator

    coordinator_address = coordinator.get_primary_endpoint()
    print("old coordinator address: {}".format(coordinator_address), file=sys.stderr)
    parsed_coordinator_address = urlparse(f"schema://{coordinator_address}")
    # because of overlay problems
    if parsed_coordinator_address.hostname == socket.gethostname():
        coordinator_address = f"127.0.0.1:{parsed_coordinator_address.port}"
        print("new coordinator address: {}".format(coordinator_address), file=sys.stderr)

    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=False,
        verbose=True,
        init_method=f"tcp://{coordinator_address}",
        rank=coordinator.get_self_index(),
        world_size=coordinator.get_total_peer_count(),
    )

    neox_args = NeoXArgs.consume_neox_args(
        input_args=input_args, overwrite_values=overwrite_values
    )
    neox_args.toolbox = toolbox
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    neox_args.initialize_comet()  # is initialized if comet directory is defined
    pretrain(neox_args=neox_args)


if __name__ == "__main__":
    main()
