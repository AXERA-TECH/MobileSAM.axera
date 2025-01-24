# MobileSAM


## 模型导出
- 进入 submodule 路径 `convert/MobileSAM`,运行以下命令
    ```shell
    $ python scripts/export_onnx_model.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./mobile_sam_decoder.onnx

    Loading model...
    Exporting onnx model to ./mobile_sam_decoder.onnx...
    ============== Diagnostic Run torch.onnx.export version 2.0.1+cpu ==============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

    ============== Diagnostic Run torch.onnx.export version 2.0.1+cpu ==============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

    Model has successfully been run with ONNXRuntime.
    ```
    可以看到目录下多了两个模型文件 `mobile_sam_encoder.onnx` `mobile_sam_decoder.onnx`


- 运行以下命令，剔除不需要的输出节点
  ```shell
  python onnx_edit.py 
  ```
  目录下会多出一个 `mobile_sam_decoder_sub.onnx`

- 运行以下命令，将动态shape改为固定shape
  ```
  $ onnxsim mobile_sam_decoder_sub.onnx mobile_sam_decoder_sub_sim.onnx --input-shape point_coords:1,5,2 point_labels:1,5

    WARNING: "--input-shape" is renamed to "--overwrite-input-shape". Please use it instead.
    Your model contains "Tile" ops or/and "ConstantOfShape" ops. Folding these ops can make the simplified model much larger. If it is not expected, please specify "--no-large-tensor" (which will lose some optimization 
    chances)
    Simplifying...
    Finish! Here is the difference:
    ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃                 ┃ Original Model ┃ Simplified Model ┃
    ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ Add             │ 83             │ 83               │
    │ Cast            │ 48             │ 6                │
    │ Concat          │ 42             │ 3                │
    │ Constant        │ 480            │ 172              │
    │ ConstantOfShape │ 5              │ 0                │
    │ Conv            │ 3              │ 3                │
    │ ConvTranspose   │ 2              │ 2                │
    │ Cos             │ 2              │ 1                │
    │ Div             │ 45             │ 24               │
    │ Equal           │ 11             │ 5                │
    │ Erf             │ 4              │ 4                │
    │ Expand          │ 6              │ 1                │
    │ Gather          │ 106            │ 5                │
    │ Gemm            │ 15             │ 15               │
    │ MatMul          │ 49             │ 48               │
    │ Mul             │ 44             │ 30               │
    │ Not             │ 1              │ 1                │
    │ Pow             │ 12             │ 12               │
    │ ReduceMean      │ 24             │ 24               │
    │ Relu            │ 12             │ 12               │
    │ Reshape         │ 36             │ 33               │
    │ Shape           │ 107            │ 0                │
    │ Sin             │ 2              │ 1                │
    │ Slice           │ 3              │ 1                │
    │ Softmax         │ 7              │ 7                │
    │ Split           │ 4              │ 0                │
    │ Sqrt            │ 12             │ 12               │
    │ Sub             │ 14             │ 14               │
    │ Transpose       │ 32             │ 30               │
    │ Unsqueeze       │ 103            │ 6                │
    │ Where           │ 5              │ 0                │
    │ Model Size      │ 15.8MiB        │ 19.6MiB          │
    └─────────────────┴────────────────┴──────────────────┘
  ```

  目录下会多出一个 `mobile_sam_decoder_sub_sim.onnx`

  以上命令都在 `export.sh` 脚本文件内记录。

## 运行
都可以在 main.py 上修改点或框的坐标来得到其他图片的结果
### PC
返回到项目根目录，编辑修改 `python_onnx/main.py`，将 encoder 和 decoder 都修改成对应的路径。
运行以下命令，得到 mask 图片
```
python python_onnx/main.py images/test.jpg 
```
上下分别为point prompt和bbox prompt的结果
![](point_mask.jpg)
![](box_mask.jpg)

### 板端
返回到项目根目录，编辑修改 `python_ax/main.py`，将 encoder 和 decoder 都修改成对应的路径。
运行以下命令，得到 mask 图片
```
python python_ax/main.py images/test.jpg 
```

### Latency
| model |resolution| latency(ms) |
|---|---|---|
|mobilesam encoder tinyvit|1024*1024|50|


## 技术讨论

- Github issues
- QQ 群: 139953715