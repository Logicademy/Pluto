# Arithmetic Operations
add x1, x2, x3          # x1 = x2 + x3
addi x4, x5, -123       # x4 = x5 + (-123)
sub x20, x21, x22       # x20 = x21 - x22
mul x3, x4, x5          # x3 = x4 * x5
mulh x6, x7, x8         # x6 = (x7 * x8) >> 32 (signed high)
mulhsu x9, x10, x11     # x9 = (x10 * x11) >> 32 (signed * unsigned)
mulhu x12, x13, x14     # x12 = (x13 * x14) >> 32 (unsigned high)
div x24, x25, x26       # x24 = x25 / x26 (signed)
divu x27, x28, x29      # x27 = x28 / x29 (unsigned)
rem x20, x21, x22       # x20 = x21 % x22 (signed)
remu x23, x24, x25      # x23 = x24 % x25 (unsigned)

# Logical Operations
and x8, x9, x10         # x8 = x9 & x10
andi x6, x7, 0x1F       # x6 = x7 & 0x1F
or x17, x18, x19        # x17 = x18 | x19
ori x15, x16, 0b1010    # x15 = x16 | 0b1010
xor x23, x24, x25       # x23 = x24 ^ x25
xori x26, x27, 0xFF     # x26 = x27 ^ 0xFF

# Shift Operations
sll x26, x27, x28       # x26 = x27 << (x28 & 0x1F)
slli x29, x30, 5        # x29 = x30 << 5
sra x10, x11, x12       # x10 = x11 >> (x12 & 0x1F) (arithmetic)
srai x13, x14, 3        # x13 = x14 >> 3 (arithmetic)
srl x15, x16, x17       # x15 = x16 >> (x17 & 0x1F) (logical)
srli x18, x19, 4        # x18 = x19 >> 4 (logical)

# Comparison Operations
slt x31, x1, x2         # x31 = (x1 < x2) ? 1 : 0
slti x3, x4, 10         # x3 = (x4 < 10) ? 1 : 0
sltiu x5, x6, -1        # x5 = (x6 < -1) ? 1 : 0 (unsigned)
sltu x7, x8, x9         # x7 = (x8 < x9, unsigned) ? 1 : 0

# Branch Operations
beq x12, x13, label1    # if (x12 == x13) jump to label1
bge x14, x15, label2    # if (x14 >= x15) jump to label2
bgeu x16, x17, label3   # if (x16 >= x17, unsigned) jump to label3
blt x18, x19, label4    # if (x18 < x19) jump to label4
bltu x20, x21, label5   # if (x20 < x21, unsigned) jump to label5
bne x22, x23, label6    # if (x22 != x23) jump to label6

# Jump and Link
jal x30, label7         # x30 = PC + 4, jump to label7
jalr x31, x1, 0x10      # x31 = PC + 4, jump to x1 + 0x10
auipc x11, 0x1000       # x11 = PC + 0x1000
lui x2, 3000            # x2 = 0xF0000 << 12

# Pseudo-Instructions
beqz x1, label1         # if (x1 == 0) jump to label1
bgez x2, label2         # if (x2 >= 0) jump to label2
bgt x3, x4, label3      # if (x3 > x4) jump to label3
bgtu x5, x6, label4     # if (x5 > x6, unsigned) jump to label4
bgtz x7, label5         # if (x7 > 0) jump to label5
ble x8, x9, label6      # if (x8 <= x9) jump to label6
bleu x10, x11, label7   # if (x10 <= x11, unsigned) jump to label7
blez x12, label8        # if (x12 <= 0) jump to label8
bltz x13, label9        # if (x13 < 0) jump to label9
bnez x14, label10       # if (x14 != 0) jump to label10
j label11               # jump to label11
jal label12             # jump to label12
jalr x15                # jump to address in x15
jr x16                  # jump to address in x16
li x17, 0x1234          # load immediate value 0x1234 into x17
mv x18, x19             # move x19 to x18
neg x20, x21            # negate x21 and store result in x20
nop                     # no operation
not x22, x23            # bitwise NOT of x23, store in x22
ret                     # return from function
seq x24, x25, x26       # x24 = (x25 == x26) ? 1 : 0
seqz x27, x28           # x27 = (x28 == 0) ? 1 : 0
sge x29, x30, x31       # x29 = (x30 >= x31) ? 1 : 0
sgt x1, x2, x3          # x1 = (x2 > x3) ? 1 : 0
sgtz x4, x5             # x4 = (x5 > 0) ? 1 : 0
sle x6, x7, x8          # x6 = (x7 <= x8) ? 1 : 0
sltz x9, x10            # x9 = (x10 < 0) ? 1 : 0
sne x11, x12, x13       # x11 = (x12 != x13) ? 1 : 0
snez x14, x15           # x14 = (x15 != 0) ? 1 : 0

# Relocation Instructions
jal x1, label13         # jump to label13 and link in x1
call label14            # call subroutine at label14
lw x1, label15          # load word from address of label15
sw t0, label16(t1)      # store word at address label16(t1)
tail label17            # jump to label17 without link

# Labels
label1:
label2:
label3:
label4:
label5:
label6:
label7:
label8:
label9:
label10:
label11:
label12:
label13:
label14:
label15:
label16:
label17:

.asciiz "hello"