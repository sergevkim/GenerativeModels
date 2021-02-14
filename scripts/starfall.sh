#!/bin/sh

name="$1"
sed -i "s/protostar/$name/g" \
    protostar/datamodules/protostar_datamodule.py \
    protostar/models/protostar_model.py \
    protostar/trainer/trainer.py \
    scripts/install_module.sh \
    setup.py
mv protostar $name

