import pytest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from tests.assembler import Assembler, Linker

@pytest.fixture
def riscv_code():
    """Returns the RISC-V assembly code for testing."""
    return """    .data                            # Data section for variables
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
    ret                              # Function return"""


@pytest.fixture
def expected_machine_code():
    """Returns the expected machine code corresponding to the RISC-V assembly code."""
    return """10000517\n00050513\n00400893\n00000073\n00500293\n00a00313\n006283b3\n406383b3\n0062feb3\n0062e7b3\n0062c833\n00629933\n005352b3\n00038463\n00000013\n01000413\n12829c63\n10000497\nfca48483\n10000517\nfc352503\n00748023\n00f52023\n10078e63\nfc081ee3\nff600593\n00f00613\n00c58863\n00859663\n00b44463\n10b45063\nfc0280e3\nfa02dee3\nfa534ce3\nfa536ae3\nfa6048e3\nfa5356e3\nfa5374e3\nfa6052e3\nfa02c0e3\n0c081c63\n00000317\n0e0300e7\n0d8000ef\n000400e7\nff9ff06f\n00008067\n10000297\nf4428293\n10000497\nf4a48483\n10000497\nf424c483\n10000497\nf3a49483\n10000497\nf324d483\n06400293\n10000517\nf2752503\n00048513\n40900533\nfff4c393\n00008067\n10000017\nf0900723\n00153493\n00a024b3\n10000017\nee901fa3\n000524b3\n00a034b3\n10000017\nee9027a3\n00000317\n05c30067\n407504b3\n0014b493\n007524b3\n0014c493\n007534b3\n0014c493\n00a3a4b3\n00a3b4b3\n00a3a4b3\n0014c493\n00a3b4b3\n0014c493\n407504b3\n009034b3\nffb48293\n02a284b3\n02b4c533\n100005b7\n00a00893\n00000513\n00000073\n00008067""".split("\n")

@pytest.fixture
def assemble_and_link():
    def inner(code):
        output = Assembler.assemble(code)
        assert not output.errors, f"Errors encountered during assembly: {output.errors}"
        linked = Linker.link([output.prog])
        assert not linked.errors, f"Errors encountered during linking: {linked.errors}"
        return linked
    return inner

def assert_machine_code(linked, expected_result):
    for index, instruction in enumerate(linked.prog.insts):
        generated_machine_code = hex(int(instruction.__str__()))[2:].zfill(8)
        assert generated_machine_code == expected_result[index], f"Mismatch at instruction {index}: expected {expected_result[index]}, got {generated_machine_code}"

def test_assembling_full_program(riscv_code, expected_machine_code, assemble_and_link):
    linked = assemble_and_link(riscv_code)
    assert_machine_code(linked, expected_machine_code)

def test_assemble_jal(assemble_and_link):
    code = """jal x1, func\nfunc:\nret"""
    expected_result = """004000ef\n00008067""".split("\n")
    linked = assemble_and_link(code)
    assert_machine_code(linked, expected_result)

def test_assemble_jalr(assemble_and_link):
    code = """jalr x1, x8, 0"""
    expected_result = """000400e7""".split("\n")
    linked = assemble_and_link(code)
    assert_machine_code(linked, expected_result)

def test_assemble_j(assemble_and_link):
    code = """loop:\nj loop"""
    expected_result = """0000006f""".split("\n")
    linked = assemble_and_link(code)
    assert_machine_code(linked, expected_result)

def test_assemble_jr(assemble_and_link):
    code = """jr ra"""
    expected_result = """00008067""".split("\n")
    linked = assemble_and_link(code)
    assert_machine_code(linked, expected_result)

def test_assemble_la(assemble_and_link):
    code = """msg:\n.asciiz "Hello, World!"\nla x5, msg"""
    expected_result = """00000297\n00028293""".split("\n")
    linked = assemble_and_link(code)
    assert_machine_code(linked, expected_result)