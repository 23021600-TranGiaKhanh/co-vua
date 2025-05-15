#—————————————————————————————
# Makefile cho Chess Engine C++
#—————————————————————————————

# Trình biên dịch và flags
CXX       := g++
CXXFLAGS  := -std=c++17 -O3 -march=native -Wall -Wextra

# Thư mục chứa mã nguồn và thư mục build obj
SRCDIR    := .
BUILDDIR  := build

# Tìm tất cả các file .cpp trong SRCDIR
SOURCES   := $(wildcard $(SRCDIR)/*.cpp)
# Tạo danh sách file .o tương ứng trong thư mục build
OBJECTS   := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))

# Tên file thực thi
TARGET    := myengine

# Mặc định: build target
all: prepare $(BUILDDIR)/$(TARGET)

# Tạo thư mục build nếu chưa có
prepare:
	@mkdir -p $(BUILDDIR)

# Rule để link từ .o thành executable
$(BUILDDIR)/$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Pattern rule: compile mỗi .cpp → .o
# $< là source, $@ là target
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Xóa toàn bộ object và binary
.PHONY: clean
clean:
	@rm -rf $(BUILDDIR)
