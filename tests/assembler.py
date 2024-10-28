# Imports
from enum import Enum
from abc import ABC, abstractmethod
from typing import List
import json

class AssemblerError(Exception):
    def __init__(self, msg=None, error_line=None, original_error=None):
        super().__init__(msg or (original_error and str(original_error)))
        self.line = error_line

    def __str__(self):
        if self.line is None:
            return super().__str__()
        return f"Line {super().__str__()}: {self.line}"

class DebugInfo:
    def __init__(self, line_no: int, line: str):
        self.line_no = line_no
        self.line = line

class RelocationInfo:
    def __init__(self, relocator, offset: int, label: str):
        self.relocator = relocator
        self.offset = offset
        self.label = label

    def __repr__(self):
        return (f"RelocationInfo(relocator={self.relocator!r}, "
                f"offset={self.offset!r}, label={self.label!r})")

class Program:
    """Represents an (unlinked) program."""

    def __init__(self, name="anonymous"):
        self.name = name
        self.insts = []  # List to store MachineCode objects
        self.debugInfo = []  # List to store DebugInfo objects
        self.labels = {}  # Dictionary to store labels and their offsets
        self.relocationTable = []  # List to store RelocationInfo objects
        self.dataSegment = bytearray()  # List to store data segment bytes
        self.textSize = 0
        self.dataSize = 0
        self.globalLabels = set()  # Set to store global labels

    def add(self, mcode):
        """Adds an instruction to the program and increments the text size."""
        self.insts.append(mcode)
        self.textSize += mcode.length

    def add_to_data(self, byte):
        """Adds a byte of data to the program and increments the data size."""
        self.dataSegment.append(byte)
        self.dataSize += 1

    def add_debug_info(self, dbg):
        """Adds debug info to the current instruction."""
        while len(self.debugInfo) < len(self.insts):
            self.debugInfo.append(dbg)

    def add_label(self, label, offset):
        """Adds a label with a given offset to the program."""
        if label in self.labels:
            raise AssemblerError(f"Label \"{label}\" defined twice")
        self.labels[label] = offset

    def get_label_offset(self, label):
        """
        Gets the relative label offset or returns None if it does not exist.

        The relative offset is calculated relative to the current text size.
        """
        loc = self.labels.get(label)
        return (loc - self.textSize) if loc is not None else None

    def add_relocation(self, relocator, label, offset=None):
        """
        Adds a line to the relocation table.

        By default, the offset is set to the current text size.
        """
        
        if offset is None:
            offset = self.textSize

        self.relocationTable.append(RelocationInfo(relocator, offset, label))

    def make_label_global(self, label):
        """Marks a label as global."""
        self.globalLabels.add(label)

    def is_global_label(self, label):
        """Checks if a label is global."""
        return label in self.globalLabels

    def dump(self):
        """Dumps the instructions."""
        return self.insts

class ProgramDebugInfo:
    def __init__(self, program_name: str, dbg):
        self.program_name = program_name
        self.dbg = dbg

    def __repr__(self):
        return f"ProgramDebugInfo(program_name={self.program_name!r}, dbg={self.dbg!r})"

class LinkedProgram:
    def __init__(self):
        self.prog = Program()
        self.errors =[]
        self.dbg = []
        self.start_pc = None

    def __repr__(self):
        return (f"LinkedProgram(prog={self.prog!r}, "
                f"dbg={self.dbg!r}, start_pc={self.start_pc!r})")

class InstructionField(Enum):
    ENTIRE = (0, 32)
    OPCODE = (0, 7)
    RD = (7, 12)
    FUNCT3 = (12, 15)
    RS1 = (15, 20)
    RS2 = (20, 25)
    FUNCT7 = (25, 32)
    IMM_11_0 = (20, 32)
    IMM_4_0 = (7, 12)
    IMM_11_5 = (25, 32)
    IMM_11_B = (7, 8)
    IMM_4_1 = (8, 12)
    IMM_10_5 = (25, 31)
    IMM_12 = (31, 32)
    IMM_31_12 = (12, 32)
    IMM_19_12 = (12, 20)
    IMM_11_J = (20, 21)
    IMM_10_1 = (21, 31)
    IMM_20 = (31, 32)
    SHAMT = (20, 25)

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi

    def get_lo(self) -> int:
        return self.lo

    def get_hi(self) -> int:
        return self.hi

class MachineCode:
    def __init__(self, encoding: int):
        self.encoding = encoding
        self.length = 4

    def __getitem__(self, ifield):
        lo = ifield.get_lo()
        hi = ifield.get_hi()
        mask = ((1 << hi) - (1 << lo))
        return (self.encoding & mask) >> lo

    def __setitem__(self, ifield, value: int):
        lo = ifield.get_lo()
        hi = ifield.get_hi()
        mask = ((1 << hi) - (1 << lo))
        self.encoding = self.encoding & ~mask
        self.encoding = self.encoding | ((value << lo) & mask)

    def __str__(self):
        return str(self.encoding)

class MemorySegments:
    STACK_BEGIN = 0x7fff_fff0
    HEAP_BEGIN = 0x1000_8000
    STATIC_BEGIN = 0x1000_0000
    TEXT_BEGIN = 0x0000_0000

class Linker:
    @staticmethod
    def link(progs):
        linked_program = LinkedProgram()
        global_table = {}
        to_relocate = []
        text_total_offset = 0
        data_total_offset = 0
        
        try:
            for prog in progs:
                for label, offset in prog.labels.items():
                    start = data_total_offset if offset >= MemorySegments.STATIC_BEGIN else text_total_offset
                    location = start + offset

                    if prog.is_global_label(label):
                        if label in global_table:
                            raise AssemblerError(f"Label \"{label}\" defined global in two different files")
                        global_table[label] = location
                        if label == "main":
                            linked_program.start_pc = location

                linked_program.prog.insts.extend(prog.insts)
                linked_program.dbg.extend([ProgramDebugInfo(prog.name, dbg) for dbg in prog.debugInfo])
                linked_program.prog.dataSegment.extend(prog.dataSegment)

                for relocated_obj in prog.relocationTable:
                    relocator = relocated_obj.relocator
                    offset = relocated_obj.offset
                    label = relocated_obj.label

                    to_address = prog.labels.get(label)
                    location = text_total_offset + offset
                    if to_address is not None:
                        mcode = linked_program.prog.insts[location // 4]
                        relocator(mcode, location, to_address)
                    else:
                        to_relocate.append(RelocationInfo(relocator, location, label))

                text_total_offset += prog.textSize
                data_total_offset += prog.dataSize

            try:
                for relocator, offset, label in to_relocate:
                    to_address = global_table.get(label)
                    if to_address is None:
                        raise AssemblerError(f"Label \"{label}\" used but not defined")
                    mcode = linked_program.prog.insts[offset // 4]
                    relocator(mcode, offset, to_address)
            except:
                raise AssemblerError(f"Label \"{label}\" used but not defined")           

        except AssemblerError as e:
                linked_program.errors.append(AssemblerError(e))
        return linked_program

class Lexer:
    @staticmethod
    def add_nonempty_word(previous: list, next_word: str):
        word = next_word.strip()
        if word:
            previous.append(word)

    @staticmethod
    def lex_line(line: str):
        """
        Lex a line into a label (if there) and a list of arguments.

        :param line: the line to lex
        :return: a tuple containing the label and tokens
        """
        current_word = []
        previous_words = []
        labels = []
        escaped = False
        in_character = False
        in_string = False
        found_comment = False

        for ch in line:
            was_delimiter = False
            was_label = False

            if ch == '#':
                found_comment = not in_string and not in_character
            elif ch == '\'':
                in_character = escaped == in_character and not in_string
            elif ch == '"':
                in_string = escaped == in_string and not in_character
            elif ch == ':':
                if not in_string and not in_character:
                    was_label = True
                    if previous_words:
                        raise AssemblerError(f"Label \"{''.join(current_word)}\" in the middle of an instruction")
            elif ch in ' \t(),':
                was_delimiter = not in_string and not in_character

            escaped = not escaped and ch == '\\'

            if found_comment:
                break

            if was_delimiter:
                Lexer.add_nonempty_word(previous_words, ''.join(current_word))
                current_word = []
            elif was_label:
                Lexer.add_nonempty_word(labels, ''.join(current_word))
                current_word = []
            else:
                current_word.append(ch)

        Lexer.add_nonempty_word(previous_words, ''.join(current_word))

        return labels, previous_words

def user_string_to_int(s: str) -> int:
    if is_character_literal(s):
        return character_literal_to_int(s)

    radix = 10
    if s.startswith("0x"):
        radix = 16
    elif s.startswith("0b"):
        radix = 2
    elif s[1:].startswith("0x"):
        radix = 16
    elif s[1:].startswith("0b"):
        radix = 2
    else:
        return int(s)

    skip_sign = 1 if s[0] in ['+', '-'] else 0
    no_radix_string = s[:skip_sign] + s[skip_sign+2:]
    return int(no_radix_string, radix)


def is_character_literal(s: str) -> bool:
    return s[0] == "'" and s[-1] == "'"

def character_literal_to_int(s: str) -> int:
    strip_single_quotes = s[1:-1]
    if strip_single_quotes == "\\'":
        return ord("'")
    if strip_single_quotes == "\"":
        return ord('"')

    try:
        parsed = bytes(strip_single_quotes, "utf-8").decode("unicode_escape")
        if len(parsed) == 0:
            raise ValueError(f"character literal {s} is empty")
        if len(parsed) > 1:
            raise ValueError(f"character literal {s} too long")
        return ord(parsed[0])
    except Exception as e:
        raise ValueError(f"could not parse character literal {s}: {str(e)}")
    
def get_immediate(s: str, min_val: int, max_val: int) -> int:
    try:
        imm = user_string_to_int(s)
    except ValueError:
        hint = " (might be too large)" if len(s) > 4 else ""
        raise AssemblerError(f"Invalid number, got {s}{hint}")

    if not (min_val <= imm <= max_val):
        raise AssemblerError(f"Immediate {s} (= {imm}) out of range (should be between {min_val} and {max_val})")

    return imm

def check_args_length(args_size: int, required: int):
    if args_size != required:
        raise AssemblerError(f"Recieved {args_size} arguments but expected {required}")
    
def check_pseudos_args_length(args: list[str], required: int):
    if len(args) != required:
        raise AssemblerError("Wrong number of arguments")

def reg_name_to_number(reg: str) -> int:
    if reg.startswith("x"):
        try:
            ret = int(reg[1:])
            if 0 <= ret <= 31:
                return ret
            raise AssemblerError(f"Register {reg} not recognized")
        except ValueError:
            raise AssemblerError(f"Register {reg} not recognized")
    
    reg_map = {
        "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
        "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8,
        "s1": 9, "a0": 10, "a1": 11, "a2": 12, "a3": 13,
        "a4": 14, "a5": 15, "a6": 16, "a7": 17, "s2": 18,
        "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23,
        "s8": 24, "s9": 25, "s10": 26, "s11": 27, "t3": 28,
        "t4": 29, "t5": 30, "t6": 31
    }
    
    if reg in reg_map:
        return reg_map[reg]
    
    raise AssemblerError(f"Register {reg} not recognized")

class FieldEqual:
    def __init__(self, ifield, required: int):
        self.ifield = ifield
        self.required = required

class InstructionFormat:
    def __init__(self, length: int, ifields):
        self.length = length
        self.ifields = ifields

    def matches(self, mcode) -> bool:
        return all(mcode[self.ifield] == self.required for field in self.ifields for self.ifield, self.required in [(field.ifield, field.required)])

    def fill(self):
        mcode = MachineCode(0)
        for field in self.ifields:
            mcode[field.ifield] = field.required
        return mcode

class OpcodeFunct3Format(InstructionFormat):
    def __init__(self, opcode: int, funct3: int):
        super().__init__(4, [
            FieldEqual(InstructionField.OPCODE, opcode),
            FieldEqual(InstructionField.FUNCT3, funct3)
        ])

class OpcodeFormat(InstructionFormat):
    def __init__(self, opcode: int):
        super().__init__(4, [FieldEqual(InstructionField.OPCODE, opcode)])

class BTypeFormat(OpcodeFunct3Format):
    def __init__(self, opcode: int, funct3: int):
        super().__init__(opcode, funct3)

class ITypeFormat(OpcodeFunct3Format):
    def __init__(self, opcode: int, funct3: int):
        super().__init__(opcode, funct3)

class RTypeFormat(InstructionFormat):
    def __init__(self, opcode: int, funct3: int, funct7: int):
        super().__init__(4, [
            FieldEqual(InstructionField.OPCODE, opcode),
            FieldEqual(InstructionField.FUNCT3, funct3),
            FieldEqual(InstructionField.FUNCT7, funct7)
        ])

class STypeFormat(OpcodeFunct3Format):
    def __init__(self, opcode: int, funct3: int):
        super().__init__(opcode, funct3)

class UTypeFormat(OpcodeFormat):
    def __init__(self, opcode: int):
        super().__init__(opcode)

class ParserError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InstructionParser(ABC):
    @abstractmethod
    def invoke(prog: Program, mcode: MachineCode, args: list):
        pass

class BTypeParser:
    B_TYPE_MIN = -2048
    B_TYPE_MAX = 2047

    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 3)

        mcode[InstructionField.RS1] = reg_name_to_number(args[0])
        mcode[InstructionField.RS2] = reg_name_to_number(args[1])

        label = args[2]
        imm = prog.get_label_offset(label)
        if imm is None:
            raise AssemblerError(f"Could not find label \"{label}\"")
        if not (BTypeParser.B_TYPE_MIN <= imm <= BTypeParser.B_TYPE_MAX):
            raise AssemblerError(f"Branch to \"{label}\" too far")

        mcode[InstructionField.IMM_11_B] = imm >> 11
        mcode[InstructionField.IMM_4_1] = imm >> 1
        mcode[InstructionField.IMM_12] = imm >> 12
        mcode[InstructionField.IMM_10_5] = imm >> 5

class DoNothingParser:
    B_TYPE_MIN = -2048
    B_TYPE_MAX = 2047

    @staticmethod
    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 0)

class ITypeParser:
    I_TYPE_MIN = -2048
    I_TYPE_MAX = 2047

    @staticmethod
    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 3)

        mcode[InstructionField.RD] = reg_name_to_number(args[0])
        mcode[InstructionField.RS1] = reg_name_to_number(args[1])
        mcode[InstructionField.IMM_11_0] = get_immediate(args[2], ITypeParser.I_TYPE_MIN, ITypeParser.I_TYPE_MAX)

class LoadParser:
    I_TYPE_MIN = -2048
    I_TYPE_MAX = 2047

    @staticmethod
    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 3)

        mcode[InstructionField.RD] = reg_name_to_number(args[0])
        mcode[InstructionField.RS1] = reg_name_to_number(args[2])
        mcode[InstructionField.IMM_11_0] = get_immediate(args[1], LoadParser.I_TYPE_MIN, LoadParser.I_TYPE_MAX)

class JALRParser:
    def invoke(prog: Program, mcode: MachineCode, args: list):
        try:
            ITypeParser().invoke(prog, mcode, args)
        except AssemblerError:
            try:
                LoadParser().invoke(prog, mcode, args)
            except AssemblerError as e_two:
                raise AssemblerError(f"Failed to parse JALR: {str(e_two)}")

class RawParser(InstructionParser):
    def __init__(self, eval):
        self.eval = eval

    def invoke(self, prog: Program, mcode: MachineCode, args: List[str]):
        self.eval(prog, mcode, args)

class RTypeParser(InstructionParser):
    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 3)

        mcode[InstructionField.RD] = reg_name_to_number(args[0])
        mcode[InstructionField.RS1] = reg_name_to_number(args[1])
        mcode[InstructionField.RS2] = reg_name_to_number(args[2])

class ShiftImmediateParser(InstructionParser):
    SHIFT_MIN = 0
    SHIFT_MAX = 31

    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 3)

        mcode[InstructionField.RD] = reg_name_to_number(args[0])
        mcode[InstructionField.RS1] = reg_name_to_number(args[1])
        mcode[InstructionField.SHAMT] = get_immediate(args[2], ShiftImmediateParser.SHIFT_MIN, ShiftImmediateParser.SHIFT_MAX)

class STypeParser(InstructionParser):
    S_TYPE_MIN = -2048
    S_TYPE_MAX = 2047

    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 3)

        imm = get_immediate(args[1], STypeParser.S_TYPE_MIN, STypeParser.S_TYPE_MAX)
        mcode[InstructionField.RS1] = reg_name_to_number(args[2])
        mcode[InstructionField.RS2] = reg_name_to_number(args[0])
        mcode[InstructionField.IMM_4_0] = imm
        mcode[InstructionField.IMM_11_5] = imm >> 5

class UTypeParser(InstructionParser):
    U_TYPE_MIN = 0
    U_TYPE_MAX = 1048575

    def invoke(prog: Program, mcode: MachineCode, args: list):
        check_args_length(len(args), 2)

        mcode[InstructionField.RD] = reg_name_to_number(args[0])
        mcode[InstructionField.IMM_31_12] = get_immediate(args[1], UTypeParser.U_TYPE_MIN, UTypeParser.U_TYPE_MAX)

class JALRelocator:
    @staticmethod
    def __call__(mcode: MachineCode, pc: int, target: int):
        imm = target - pc
        mcode[InstructionField.IMM_20] = imm >> 20
        mcode[InstructionField.IMM_10_1] = imm >> 1
        mcode[InstructionField.IMM_19_12] = imm >> 12
        mcode[InstructionField.IMM_11_J] = imm >> 11

class PCRelHiRelocator:
    @staticmethod
    def __call__(mcode: MachineCode, pc: int, target: int):
        imm = (target - pc + 0x800) >> 12
        mcode[InstructionField.IMM_31_12] = imm

class PCRelLoRelocator:
    @staticmethod
    def __call__(mcode: MachineCode, pc: int, target: int):
        mcode[InstructionField.IMM_11_0] = target - (pc - 4)

class PCRelLoStoreRelocator:
    @staticmethod
    def __call__(mcode: MachineCode, pc: int, target: int):
        offset = target - (pc - 4)
        mcode[InstructionField.IMM_4_0] = offset
        mcode[InstructionField.IMM_11_5] = offset >> 5

class Relocator:
    def __init__(self, relocator):
        self.relocator = relocator

    def __call__(self, mcode, pc: int, target: int):
        self.relocator(mcode, pc, target)

JALRelocator = Relocator(JALRelocator())
PCRelHiRelocator = Relocator(PCRelHiRelocator())
PCRelLoRelocator = Relocator(PCRelLoRelocator())
PCRelLoStoreRelocator = Relocator(PCRelLoStoreRelocator())

class Instruction:
    all_instructions = []

    def __init__(self, name: str, format: InstructionFormat, parser: InstructionParser):
        self.name = name
        self.format = format
        self.parser = parser
        Instruction.all_instructions.append(self)

    @staticmethod
    def get_by_name(name: str) -> 'Instruction':
        for instruction in Instruction.all_instructions:
            if instruction.name == name:
                return instruction
        raise AssemblerError(f"Instruction with name \"{name}\" not found")

    def __str__(self) -> str:
        return self.name
    
class BTypeInstruction(Instruction):
    def __init__(self, name: str, opcode: int, funct3: int):
       
        super().__init__(
            name=name,
            format=BTypeFormat(opcode, funct3),
            parser=BTypeParser
        )

class ITypeInstruction(Instruction):
    def __init__(self, name: str, opcode: int, funct3: int):  
        super().__init__(
            name=name,
            format=ITypeFormat(opcode, funct3),
            parser=ITypeParser
        )

class LoadTypeInstruction(Instruction):
    def __init__(self, name: str, opcode: int, funct3: int):
        super().__init__(
            name=name,
            format=ITypeFormat(opcode, funct3),
            parser=LoadParser
        )

class RTypeInstruction(Instruction):
    def __init__(self, name: str, opcode: int, funct3: int, funct7: int):

        super().__init__(
            name=name,
            format=RTypeFormat(opcode, funct3, funct7),
            parser=RTypeParser
        )

class ShiftImmediateInstruction(Instruction):
    def __init__(self, name: str, funct3: int, funct7: int):

        super().__init__(
            name=name,
            format=RTypeFormat(opcode=0b0010011, funct3=funct3, funct7=funct7),
            parser=ShiftImmediateParser
        )

class STypeInstruction(Instruction):
    def __init__(self, name: str, opcode: int, funct3: int):
        super().__init__(
            name=name,
            format=STypeFormat(opcode, funct3),
            parser=STypeParser
        )

class UTypeInstruction(Instruction):
    def __init__(self, name: str, opcode: int):
        super().__init__(
            name=name,
            format=UTypeFormat(opcode),
            parser=UTypeParser
        )

def jal_parser(prog, mcode, args):
    check_args_length(len(args), 2)
    mcode[InstructionField.RD] = reg_name_to_number(args[0]) 
    prog.add_relocation(JALRelocator, args[1])

def create_utype_instruction(name: str, opcode: int): return UTypeInstruction(name=name, opcode=opcode)
def create_stype_instruction(name: str, funct3: int): return STypeInstruction(name=name, opcode=0b0100011, funct3=funct3)
def create_btype_instruction(name: str, funct3: int): return BTypeInstruction(name=name, opcode=0b1100011, funct3=funct3)
def create_itype_instruction(name: str, funct3: int): return ITypeInstruction(name=name, opcode=0b0010011, funct3=funct3)
def create_loadtype_instruction(name: str, funct3: int): return LoadTypeInstruction(name=name, opcode=0b0000011, funct3=funct3)
def create_rtype_instruction(name: str, funct3: int, funct7: int): return RTypeInstruction(name=name, opcode=0b0110011, funct3=funct3, funct7=funct7)
def create_shift_immediate_instruction(name: str, funct3: int, funct7: int): return ShiftImmediateInstruction(name=name, funct3=funct3, funct7=funct7)

jal = Instruction(name="jal", format=OpcodeFormat(0b1101111), parser=RawParser(jal_parser))
jalr = Instruction(name="jalr", format=ITypeFormat(opcode=0b1100111, funct3=0b000), parser=JALRParser)
ecall = Instruction(name="ecall",format=InstructionFormat(length=4, ifields=[FieldEqual(InstructionField.ENTIRE, 0b000000000000_00000_000_00000_1110011)]), parser=DoNothingParser())

add_instruction = create_rtype_instruction("add", 0b000, 0b0000000)
and_instruction = create_rtype_instruction("and", 0b111, 0b0000000)
div_instruction = create_rtype_instruction("div", 0b100, 0b0000001)
divu_instruction = create_rtype_instruction("divu", 0b101, 0b0000001)
mul_instruction = create_rtype_instruction("mul", 0b000, 0b0000001)
mulh_instruction = create_rtype_instruction("mulh", 0b001, 0b0000001)
mulhsu_instruction = create_rtype_instruction("mulhsu", 0b010, 0b0000001)
mulhu_instruction = create_rtype_instruction("mulhu", 0b011, 0b0000001)
or_instr_instruction = create_rtype_instruction("or", 0b110, 0b0000000)
rem_instruction = create_rtype_instruction("rem", 0b110, 0b0000001)
remu_instruction = create_rtype_instruction("remu", 0b111, 0b0000001)
sll_instruction = create_rtype_instruction("sll", 0b001, 0b0000000)
slt_instruction = create_rtype_instruction("slt", 0b010, 0b0000000)
sltu_instruction = create_rtype_instruction("sltu", 0b011, 0b0000000)
sra_instruction = create_rtype_instruction("sra", 0b101, 0b0100000)
srl_instruction = create_rtype_instruction("srl", 0b101, 0b0000000)
sub_instruction = create_rtype_instruction("sub", 0b000, 0b0100000)
xor_instruction = create_rtype_instruction("xor", 0b100, 0b0000000)

addi_instruction = create_itype_instruction("addi", 0b000)
andi_instruction = create_itype_instruction("andi", 0b111)
ori_instruction = create_itype_instruction("ori", 0b110)
slti_instruction = create_itype_instruction("slti", 0b010)
sltiu_instruction = create_itype_instruction("sltiu", 0b011)
xori_instruction = create_itype_instruction("xori", 0b100)

beq_instruction = create_btype_instruction("beq", 0b000)
bge_instruction = create_btype_instruction("bge", 0b101)
bgeu_instruction = create_btype_instruction("bgeu", 0b111)
blt_instruction = create_btype_instruction("blt", 0b100)
bltu_instruction = create_btype_instruction("bltu", 0b110)
bne_instruction = create_btype_instruction("bne", 0b001)

lb_instruction = create_loadtype_instruction("lb", 0b000)
lbu_instruction = create_loadtype_instruction("lbu", 0b100)
lh_instruction = create_loadtype_instruction("lh", 0b001)
lhu_instruction = create_loadtype_instruction("lhu", 0b101)
lw_instruction = create_loadtype_instruction("lw", 0b010)

slli_instruction = create_shift_immediate_instruction("slli", 0b001, 0b0000000)
srai_instruction = create_shift_immediate_instruction("srai", 0b101, 0b0100000)
srli_instruction = create_shift_immediate_instruction("srli", 0b101, 0b0000000)

sb_instruction = create_stype_instruction("sb", 0b000)
sh_instruction = create_stype_instruction("sh", 0b001)
sw_instruction = create_stype_instruction("sw", 0b010)

auipc_instruction = create_utype_instruction("auipc", 0b0010111)
lui_instruction = create_utype_instruction("lui", 0b0110111)

class PseudoWriter:
    def __call__(self, args, state):
        raise NotImplementedError("Subclasses should implement this method.")

class BEQZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [[ "beq", args[1], "x0", args[2] ]]

class BGEZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["bge", args[1], "x0", args[2]] ]

class BGT(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        return [ ["blt", args[2], args[1], args[3]] ]
    
class BGTU(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        return [ ["bltu", args[2], args[1], args[3]] ]
    
class BGTZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["blt", "x0", args[1], args[2]] ]

class BLE(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        return [ ["bge", args[2], args[1], args[3]] ]
    
class BLEU(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        return [ ["bgeu", args[2], args[1], args[3]] ]
    
class BLEZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["bge", "x0", args[1], args[2]] ]
    
class BLTZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["blt", args[1], "x0", args[2]] ]
    
class BNEZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["bne", args[1], "x0", args[2]] ]
    
class CALL(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 2)

        auipc = ["auipc", "x6", "0"]
        jalr = ["jalr", "x1", "x6", "0"]

        state.add_relocation(PCRelHiRelocator, state.get_offset(), args[1])
        state.add_relocation(PCRelLoRelocator, state.get_offset() + 4, args[1])

        return [auipc, jalr]

class J(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 2)
        return [ ["jal", "x0", args[1]] ]
    
class JAL(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 2)
        return [ ["jal", "x1", args[1]] ]
    
class JALR(PseudoWriter):
    def __call__(self, args, state):
        if len(args) == 2:
            return [ ["jalr", "x1", args[1], "0"] ]
        elif len(args) == 4:
            return [args]
        else:
            raise AssemblerError("Wrong number of arguments")
        
class JR(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 2)
        return [ ["jalr", "x0", args[1], "0"] ]
    
class LA(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)

        auipc = ["auipc", args[1], "0"]
        state.add_relocation(PCRelHiRelocator, state.get_offset(), args[2])

        addi = ["addi", args[1], args[1], "0"]
        state.add_relocation(PCRelLoRelocator, state.get_offset() + 4, args[2])

        return [auipc, addi]
    
class LI(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        
        try:
            imm = user_string_to_int(args[2])
        except ValueError:
            raise AssemblerError("Immediate to LI too large or NaN")

        if -2048 <= imm <= 2047:
            return [ ["addi", args[1], "x0", args[2]] ]
        else:
            imm_hi = (imm + 0x800) >> 12
            imm_lo = imm - (imm_hi << 12)
            lui = ["lui", args[1], str(imm_hi)]
            addi = ["addi", args[1], args[1], str(imm_lo)]
            return [lui, addi]
        
class Load(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)

        auipc = ["auipc", args[1], "0"]
        state.add_relocation(PCRelHiRelocator, state.get_offset(), args[2])

        load = [args[0], args[1], "0", args[1]]
        state.add_relocation(PCRelLoRelocator, state.get_offset() + 4, args[2])

        return [auipc, load]

class MV(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [[ "addi", args[1], args[2], "0" ]]
    
class NEG(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [[ "sub", args[1], "x0", args[2] ]]
    
class NOP(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 1)
        return [[ "addi", "x0", "x0", "0" ]]
    
class NOT(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [[ "xori", args[1], args[2], "-1" ]]
    
class RET(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 1)
        return [[ "jalr", "x0", "x1", "0" ]]
    
class SEQ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        subtract = ["sub", args[1], args[2], args[3]]
        check_zero = ["sltiu", args[1], args[1], "1"]
        return [subtract, check_zero]
    
class SEQZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["sltiu", args[1], args[2], "1"] ]
    
class SGE(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        unsigned = "u" if args[0].endswith("u") else ""
        set_instruction = [f"slt{unsigned}", args[1], args[2], args[3]]
        invert_instruction = ["xori", args[1], args[1], "1"]
        
        return [set_instruction, invert_instruction]

class SGT(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        unsigned = "u" if args[0].endswith("u") else ""
        sgt_instruction = [f"slt{unsigned}", args[1], args[3], args[2]]
        
        return [sgt_instruction]
    
class SGTZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        sgtz_instruction = ["slt", args[1], "x0", args[2]]
        
        return [sgtz_instruction]
    
class SLE(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        unsigned = "u" if args[0].endswith("u") else ""
        set_instruction = ["slt" + unsigned, args[1], args[3], args[2]]
        invert_instruction = ["xori", args[1], args[1], "1"]
        
        return [set_instruction, invert_instruction]
    
class SLTZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["slt", args[1], args[2], "x0"] ]
    
class SNE(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        
        subtract = ["sub", args[1], args[2], args[3]]
        check_non_zero = ["sltu", args[1], "x0", args[1]]
        
        return [subtract, check_non_zero]
    
class SNEZ(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 3)
        return [ ["sltu", args[1], "x0", args[2]] ]
    
class Store(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 4)
        label = args[2]
        try:
            user_string_to_int(label)
            return [args]
        except ValueError:
            pass

        auipc = ["auipc", args[3], "0"]
        state.add_relocation(PCRelHiRelocator, state.get_offset(), label)

        store = [args[0], args[1], "0", args[3]]
        state.add_relocation(PCRelLoStoreRelocator, state.get_offset() + 4, label)

        return [auipc, store]
    
class TAIL(PseudoWriter):
    def __call__(self, args, state):
        check_pseudos_args_length(args, 2)

        auipc = ["auipc", "x6", "0"]
        state.add_relocation(PCRelHiRelocator, state.get_offset(), args[1])

        jalr = ["jalr", "x0", "x6", "0"]
        state.add_relocation(PCRelLoRelocator, state.get_offset() + 4, args[1])

        return [auipc, jalr]

class PseudoDispatcher:
    """
    Describes each instruction for writing.
    """
    BEQZ = BEQZ
    BGEZ = BGEZ
    BGT = BGT
    BGTU = BGTU
    BGTZ = BGTZ
    BLE = BLE
    BLEU = BLEU
    BLEZ = BLEZ
    BLTZ = BLTZ
    BNEZ = BNEZ
    CALL = CALL
    JAL = JAL
    JALR = JALR
    J = J
    JR = JR
    LA = LA
    LB = Load
    LBU = Load
    LH = Load
    LHU = Load
    LI = LI
    LW = Load
    MV = MV
    NEG = NEG
    NOP = NOP
    NOT = NOT
    RET = RET
    SB = Store
    SEQZ = SEQZ
    SGTZ = SGTZ
    SH = Store
    SLTZ = SLTZ
    SNEZ = SNEZ
    SW = Store
    TAIL = TAIL
    SEQ = SEQ
    SGE = SGE
    SGEU = SGE
    SGT = SGT
    SGTU = SGT
    SLE = SLE
    SLEU = SLE
    SNE = SNE

class Assembler:
    @staticmethod
    def assemble(text: str):
        pass_one_prog, tal_instructions, pass_one_errors = AssemblerPassOne(text).run()
        if pass_one_errors:
            return AssemblerOutput(pass_one_prog, pass_one_errors, [])
        pass_two_output = AssemblerPassTwo(pass_one_prog, tal_instructions).run()
        
        return pass_two_output

class DebugInstruction:
    def __init__(self, debug: DebugInfo, line_tokens: list):
        self.debug = debug
        self.line_tokens = line_tokens


class PassOneOutput:
    def __init__(self, prog: Program, tal_instructions: list, errors: list):
        self.prog = prog
        self.tal_instructions = tal_instructions
        self.errors = errors


class AssemblerOutput:
    def __init__(self, prog: Program, errors: list, tal_instructions):
        self.prog = prog
        self.errors = errors
        self.tal_instructions = tal_instructions


class AssemblerPassOne:
    def __init__(self, text: str):
        self.text = text
        self.prog = Program()
        self.current_text_offset = MemorySegments.TEXT_BEGIN
        self.current_data_offset = MemorySegments.STATIC_BEGIN
        self.in_text_segment = True
        self.tal_instructions = []
        self.current_line_number = 0
        self.errors = []

    def run(self):
        self.do_pass_one()
        return self.prog, self.tal_instructions, self.errors

    def do_pass_one(self):
        for line in self.text.split('\n'):
            try:
                self.current_line_number += 1
                offset = self.get_offset()
                labels, args = Lexer.lex_line(line)

                for label in labels:
                    old_offset = self.prog.add_label(label, offset)
                    if old_offset is not None:
                        raise AssemblerError(f"Label \"{label}\" defined twice")

                if not args or not args[0]:
                    continue  # empty line

                if self.is_assembler_directive(args[0]):
                    self.parse_assembler_directive(args[0], args[1:], line)
                else:
                    expanded_insts = self.replace_pseudo_instructions(args)

                    for inst in expanded_insts:
                        dbg = DebugInfo(self.current_line_number, line)
                        self.tal_instructions.append(DebugInstruction(dbg, inst))
                        self.current_text_offset += 4
            except AssemblerError as e:
                self.errors.append(AssemblerError(self.current_line_number, e))

    def get_offset(self):
        return self.current_text_offset if self.in_text_segment else self.current_data_offset

    def is_assembler_directive(self, cmd: str) -> bool:
        return cmd.startswith(".")

    def replace_pseudo_instructions(self, tokens: list) -> list:
        dispatcher = PseudoDispatcher()
        cmd = get_instruction(tokens).upper()
        try:
            instruction_class = getattr(dispatcher, cmd)
            instruction_handler = instruction_class()
            return instruction_handler(tokens, self)
        except Exception as e:
            return [tokens]
            

    def parse_assembler_directive(self, directive: str, args: list, line: str):
        if directive == ".data":
            self.in_text_segment = False
        elif directive == ".text":
            self.in_text_segment = True
        elif directive == ".byte":
            for arg in args:
                byte = user_string_to_int(arg)
                if byte not in range(-127, 256):
                    raise AssemblerError(f"Invalid byte {byte} too big")
                self.prog.add_to_data(byte & 0xFF)
                self.current_data_offset += 1
        elif directive == ".asciiz":
            check_pseudos_args_length(args, 1)
            try:
                ascii_str = json.loads(args[0])
            except Exception:
                raise AssemblerError(f"Couldn't parse {args[0]} as a string")
            for c in ascii_str:
                if ord(c) not in range(0, 128):
                    raise AssemblerError(f"Unexpected non-ascii character: {c}")
                self.prog.add_to_data(ord(c))
                self.current_data_offset += 1
            self.prog.add_to_data(0)
            self.current_data_offset += 1
        elif directive == ".word":
            for arg in args:
                word = user_string_to_int(arg)
                self.prog.add_to_data(word & 0xFF)
                self.prog.add_to_data((word >> 8) & 0xFF)
                self.prog.add_to_data((word >> 16) & 0xFF)
                self.prog.add_to_data((word >> 24) & 0xFF)
                self.current_data_offset += 4
        elif directive == ".globl":
            for label in args:
                self.prog.make_label_global(label)
        else:
            raise AssemblerError(f"Unknown assembler directive \"{directive}\"")

    def add_relocation(self, relocator, offset: int, label: str):
        self.prog.add_relocation(relocator, label, offset)


class AssemblerPassTwo:
    def __init__(self, prog: Program, tal_instructions: list):
        self.prog = prog
        self.tal_instructions = tal_instructions
        self.errors = []

    def run(self):
        for dbg_inst in self.tal_instructions:
            dbg, inst = dbg_inst.debug, dbg_inst.line_tokens
            try:
                self.add_instruction(inst)
                self.prog.add_debug_info(dbg)
            except AssemblerError as e:
                self.errors.append(AssemblerError(dbg.line_no, e))
        return AssemblerOutput(self.prog, self.errors, self.tal_instructions)

    def add_instruction(self, tokens: list):
        if not tokens or not tokens[0]:
            return
        
        cmd = get_instruction(tokens)
        inst = Instruction.get_by_name(cmd)
        mcode = inst.format.fill()
        inst.parser.invoke(self.prog, mcode, tokens[1:])
        self.prog.add(mcode)


def get_instruction(tokens: list) -> str:
    return tokens[0].lower()