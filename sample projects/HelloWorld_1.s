    .data                            # Data section for variables
msg:                                 # Label for the "Hello, World!" string
    .asciiz "Hello, World!"          # Null-terminated string

num_byte:                            # Label for a byte variable
    .byte 0x7F                       # A single byte in hexadecimal format

num_word:                            # Label for a word variable
    .word -123456                    # A word (4 bytes) in decimal format with a negative immediate

    .text                            # Text section for code
    .globl _start                    # Define the global entry point

_start:
    # Load address of msg into register a0 using pseudo-instruction
    la x10, msg                      # Load address of msg into x10 (a0)
    li x17, 4                        # Load syscall code for print_string (usually 4) into x17 (a7)
    ecall                            # Make the syscall to print "Hello, World!"

    # Test Arithmetic Instructions
    li x5, 5                         # Load immediate (decimal) 5 into x5 (t0)
    li t1, 10                        # Load immediate (decimal) 10 into t1
    add t2, x5, t1                   # Add x5 and t1, result in t2 (t2 = 15)
    sub x7, t2, t1                   # Subtract t1 from t2, result in x7 (t3 = 5)

    # Test Bitwise Instructions
    and t4, x5, t1                   # Bitwise AND of x5 and t1, result in t4
    or x15, x5, t1                   # Bitwise OR of x5 and t1, result in x15 (t5)
    xor x16, x5, t1                  # Bitwise XOR of x5 and t1, result in x16 (t6)

    # Test Shift Instructions
    sll x18, x5, t1                  # Shift left logical x5 by t1, result in x18 (t7)
    srl t0, t1, x5                   # Shift right logical t1 by x5, result in t0

    # Test Branching Instructions
    beqz x7, next                    # Branch to next if x7 is zero (should not branch)
    nop                              # No operation (testing NOP)
next:
    li x8, 0x10                      # Load immediate (hexadecimal) 0x10 into x8 (s0)
    bne t0, x8, end                  # Branch to end if t0 != x8

    # Load and Store Testing
    lb x9, num_byte                  # Load byte from num_byte into x9 (t1)
    lw x10, num_word                 # Load word from num_word into x10 (t2)
    sb x7, 0(x9)                     # Store byte from x7 to num_byte
    sw x15, 0(x10)                   # Store word from x15 to num_word

    # Additional Pseudo-instruction Testing
    beqz x15, end                    # Branch if equal to zero
    bnez x16, next                   # Branch if not equal to zero
    li x11, -0b1010                  # Load binary negative immediate into x11 (s1)
    li x12, +0b1111                  # Load binary positive immediate into x12 (s2)

    # Branch Instructions
    beq x11, x12, skip               # Branch if x11 == x12
    bne x11, x8, skip                # Branch if x11 != x8
    blt x8, x11, skip                # Branch if x8 < x11
    bge x8, x11, end                 # Branch if x8 >= x11

skip:
    # Pseudo-instruction Testing
    beqz x5, next                    # BEQZ - Branch if equal to zero
    bgez x5, next                    # BGEZ - Branch if greater than or equal to zero
    bgt x5, t1, next                 # BGT - Branch if greater than
    bgtu x5, t1, next                # BGTU - Branch if greater than unsigned
    bgtz t1, next                    # BGTZ - Branch if greater than zero
    ble x5, t1, next                 # BLE - Branch if less than or equal
    bleu x5, t1, next                # BLEU - Branch if less than or equal unsigned
    blez t1, next                    # BLEZ - Branch if less than or equal to zero
    bltz x5, next                    # BLTZ - Branch if less than zero
    bnez x16, end                    # BNEZ - Branch if not equal to zero
    call func                        # CALL - Function call

loop:
    jal x1, func                     # JAL - Jump and link to function
    jalr x1, x8, 0                   # JALR - Jump and link register
    j loop                           # J - Unconditional jump
    jr ra                            # JR - Jump to return address
    la x5, msg                       # LA - Load address
    lb x9, num_byte                  # LB - Load byte
    lbu x9, num_byte                 # LBU - Load byte unsigned
    lh x9, num_byte                  # LH - Load halfword
    lhu x9, num_byte                 # LHU - Load halfword unsigned
    li x5, 100                       # LI - Load immediate
    lw x10, num_word                 # LW - Load word
    mv x10, x9                       # MV - Move
    neg x10, x9                      # NEG - Negate
    not x7, x9                       # NOT - Bitwise NOT
    ret                              # RET - Return from function
    sb x9, num_byte x0               # SB - Store byte
    seqz x9, x10                     # SEQZ - Set if equal to zero
    sgtz x9, x10                     # SGTZ - Set if greater than zero
    sh x9, num_word x0               # SH - Store halfword
    sltz x9, x10                     # SLTZ - Set if less than zero
    snez x9, x10                     # SNEZ - Set if not equal to zero
    sw x9, num_word x0               # SW - Store word
    tail func                        # TAIL - Tail call
    seq x9, x10, x7                  # SEQ - Set if equal
    sge x9, x10, x7                  # SGE - Set if greater than or equal
    sgeu x9, x10, x7                 # SGEU - Set if greater than or equal unsigned
    sgt x9, x10, x7                  # SGT - Set if greater than
    sgtu x9, x10, x7                 # SGTU - Set if greater than unsigned
    sle x9, x10, x7                  # SLE - Set if less than or equal
    sleu x9, x10, x7                 # SLEU - Set if less than or equal unsigned
    sne x9, x10, x7                  # SNE - Set if not equal

    # R-Type and U-Type Instruction Testing
    addi x5, x9, -5                  # Add immediate -5 to x9, store in x5
    mul x9, x5, x10                  # Multiply x5 and x10, store result in x9
    div x10, x9, x11                 # Divide x9 by x11, store quotient in x10
    lui x11, 0x10000                 # Load upper immediate (testing lui)

end:
    # Exit the program
    li x17, 10                       # Load syscall code for exit (usually 10) into x17 (a7)
    li x10, 0                        # Load immediate 0 (success) into x10 (a0)
    ecall                            # Exit syscall to terminate program

func:
    ret                              # Function return