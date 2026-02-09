BUILD_DIR := build
TARGET := app

.PHONY: all configure build run clean rebuild

all: build

configure:
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

build: configure
	@cmake --build $(BUILD_DIR) -j

run: build
	@./$(BUILD_DIR)/$(TARGET)

clean:
	@rm -rf $(BUILD_DIR)

rebuild: clean build
