BUILD_DIR="build"
SRC_DIR="$BUILD_DIR/pesco"

# Clean build dir
if [ -d "$BUILD_DIR" ]; then
    echo "Delete $BUILD_DIR..."
    rm -rf $BUILD_DIR
fi

# Create folder for zip
mkdir -p "$BUILD_DIR"
mkdir -p "$SRC_DIR"

# Move everthing necessary
echo "Copy PeSCo sources"
cp -r bin/ "$SRC_DIR/bin"

mkdir -p "$SRC_DIR/pesco/pesco"
cp pesco/pesco/*.py "$SRC_DIR/pesco/pesco"

mkdir -p "$SRC_DIR/pesco/pesco/data"
cp pesco/pesco/data/*.py "$SRC_DIR/pesco/pesco/data"

cp -r properties "$SRC_DIR/properties"
cp README.md "$SRC_DIR/README.md"
cp LICENSE "$SRC_DIR/LICENSE"

mkdir -p "$SRC_DIR/lib"

# Deploy CPAchecker
echo "Copy CPAchecker"
mkdir -p "$SRC_DIR/lib/cpachecker"
cp -r lib/cpachecker/config "$SRC_DIR/lib/cpachecker/config"
cp -r lib/cpachecker/doc "$SRC_DIR/lib/cpachecker/doc"
cp -r lib/cpachecker/lib "$SRC_DIR/lib/cpachecker/lib"
cp -r lib/cpachecker/LICENSES "$SRC_DIR/lib/cpachecker/LICENSES"
cp -r lib/cpachecker/scripts "$SRC_DIR/lib/cpachecker/scripts"
cp lib/cpachecker/cpachecker.jar "$SRC_DIR/lib/cpachecker/cpachecker.jar"
cp lib/cpachecker/README.md "$SRC_DIR/lib/cpachecker/README.md"
cp lib/cpachecker/Authors.md "$SRC_DIR/lib/cpachecker/Authors.md"
cp lib/cpachecker/INSTALL.md "$SRC_DIR/lib/cpachecker/INSTALL.md"
cp lib/cpachecker/LICENSE "$SRC_DIR/lib/cpachecker/LICENSE"

# KLEE
echo "Copy KLEE"
cp -r lib/klee "$SRC_DIR/lib/klee"

# CLANG
echo "Copy CLANG"
cp -r lib/clang "$SRC_DIR/lib/clang"

# Python dependencies
echo "Copy Python dependencies"
cp -r lib/python "$SRC_DIR/lib/python"

echo "Copy Resources"
cp -r resource "$SRC_DIR/resource"

echo "Zip archive"
pushd "$BUILD_DIR"
zip -r "pesco.zip" "pesco"
popd