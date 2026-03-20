
# Minimal Makefile to make sure I have what I want

# --------------------------------------------------
# 					Inputs
# --------------------------------------------------

CC = g++
OBJDUMP = objdump

TARGET = cq
SRC += cq.cc

# Where we are saving
BUILD_DIR = build

# --------------------------------------------------
# 		Targets in order of importance
# --------------------------------------------------

# (1) Ensure build dir exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# (2) Make binary yay
$(BUILD_DIR)/$(TARGET).bin: $(SRC) | $(BUILD_DIR)
	$(CC) -o $@ $<

# (3) Generate assembly dump for checking (thanks cs140e)
$(BUILD_DIR)/%.list: $(BUILD_DIR)/%.bin
	$(OBJDUMP) -D $< > $@

# (4) The actual make command (NO WHITESPACE AFTER THE SLASH i learned that the hard way)
all: $(BUILD_DIR)/$(TARGET).bin \
     $(BUILD_DIR)/$(TARGET).list

# (5) ifff we want to run as well just do `make
run: all
	./$(BUILD_DIR)/$(TARGET).bin

# Clean
clean:
	rm -rf $(BUILD_DIR)



.PHONY: all clean