
## Glad
refer: https://github.com/libigl/libigl/blob/main/cmake/recipes/external/glad.cmake

```shell
git clone https://github.com/libigl/libigl-glad
mkdir build
cd build
cmake ..
cmake --build . --config Release
```


インクルードディレクトリ: path/to/libigl-glad/include
ライブラリディレクトリ: path/to/libigl-glad/build/src/Release
リンカーの入力:glad.lib


## GLFW3
refer: https://github.com/libigl/libigl/blob/main/cmake/recipes/external/glfw.cmake

```shell
git clone --depth 1 --branch 3.3.7 https://github.com/glfw/glfw.git
mkdir build
cd build
cmake .. -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF -DGLFW_INSTALL=OFF -DGLFW_VULKAN_STATIC=OFF # お好みで
cmake --build . --config Release
```


インクルードディレクトリ: path/to/glfw/include
ライブラリディレクトリ: path/to/glfw/build/src/Release
リンカーの入力:glfw3.lib


## MSYS2
libwinpthread-1.dllがないなどとお怒りを受けるので, MSYS2インストール時についてくるDLLを利用

以下を参考にする
https://github.com/greenfork/nimraylib_now/pull/73/files#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5

## settings
C/C++ -> コード生成 -> ランタイムライブラリ -> マルチスレッドDLL(/MD)



