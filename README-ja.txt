# インバースレンダリングを用いた画像の相互反射を考慮した色編集

## ディレクトリ構造

## 動作環境とインストール
使用するパッケージ、特にmitsubaとdr.jitのアップデートを確認してください。

condaなどの仮想環境の利用を強く推奨します。
作成した仮想環境で、以下のコマンドを実行してパッケージをインストールしてください。：
```
pip install -r requirements.txt
```
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)と[StableNormal](https://github.com/Stable-X/StableNormal)を使用する場合は、geometry_predictionの中にクローンする必要があります。


## 色編集

1. 入力データを用意する
    * png、jpg、exrの3形式に対応。
    
2. 深度マップと法線マップの推定
    * このプロジェクトでは、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)と[StableNormal](https://github.com/Stable-X/StableNormal)を使用しました。
    * より正確な手法を使用できる場合は、そちらを使用してください。

3. マテリアル推定
    * プロジェクトでは、[IndoorInverseRendering](https://github.com/jingsenzhu/IndoorInverseRendering) の mgnet を使用しました。
    * より正確な方法を使用できる場合は、それを使用してください。

4. Mitsuba 3用のポリゴンメッシュファイルを作成する。
```
python make_polygon_mesh.py --dir_path (path for input data directory) --fov (field of view) --name (mesh file name to be given) -he (height of the input image) -wi (width of the input image)
```
    * 入力ディレクトリが以下のような構造になっている場合、-dir_pathを指定することでポリゴンメッシュファイルを作成することができます。
```
.
└── example1(any name)
    ├── depth.npy
    ├── normal.png
    └── dense_v1
        ├── albedo -- 000.exr
        └── material -- 000.exr
```
    * そうでない場合、-depth_path、--albedo_path、--material_path、--normal_path を個別に設定してください。
    * StableNormal以外の方法で推定した法線マップを使用する場合は、--not_use_StableNormalを追加してください。

5. 光源の最適化。
```
python arealight_optimization.py --image_path (path for the input image) --mesh_path (path for the mesh file)  --fov (field of view) --name (scene name) -he (height of the input image)
```
    * 入力画像のシーン名を指定します。
    * その他のオプション引数は以下の通りです。条件を変更したい場合は、これらを使用してください。
        * --spp
        * --light_num
        * --fix_position
        * --size_opt
        * --use_envmap
        * --iteration_count
        * --learning_rate
    * メモリエラーが発生した場合は、-sppの値を小さくしてください。
    * 最適化後のログと結果（npyファイル）はresults/'+(--name)+'_arealight_opt_disk_results/'+(月_日_時:分:秒)にあります。

6. マスクでアルベドを編集する。
    * 0（アルベドを変化させない）または255（アルベドを変化させる）で構成されるマスク画像を用意してください。GIMPを用いると便利です。
    * アルベドを変更するたびに別のポリゴンメッシュファイルを作成する必要があります。
```
python make_polygon_mesh.py --dir_path xxx --fov n --name yyy -h H -w W --albedo_mask_path (path for the mask image) --new_albedo_value r g b 
```
    * 新しいアルベドのRGB値を0から1の間で割り当ててください。

7. 画像の再レンダリング
```
python rerender_image.py --result_path (directory path for optimization results) --image_path yyy --mesh_path_1 (path for original mesh file) --mesh_path_2 (path for albedo changed mesh file) --fov n -h H -w W 
```
    * 結果画像は --result_path に出力されます。

# 関連研究
* [IndoorLightEditing](https://github.com/ViLab-UCSD/IndoorLightEditing)
* [rgbx](https://github.com/zheng95z/rgbx)
* [IntrinsicImageDiffusion](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion)