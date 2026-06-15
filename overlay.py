from __future__ import annotations
import struct
import re
import sotn_utils.yaml_ext as yaml
import sotn_utils.mips as mips
from collections import deque
from .helpers import get_logger, get_symbol_address, sort_subsegments
from pathlib import Path
from types import SimpleNamespace

# Todo: Convert non-mutating SimpleNamespace to namedtuple
from collections import namedtuple
from typing import Union, Optional

__all__ = [
    "MwOverlayHeader",
    "get_text_offset",
    "get_bss_offset",
    "get_rodata_address",
    "find_segments",
]


class MwOverlayHeader:
    """A Python implementation of the MetroWerks overlay header"""

    # https://gist.github.com/Linblow/541a3b24559f9c89374fdbd9e0693c40
    """
    typedef struct {
    /* 0x0 */ uint8_t identifier[3];   // Header identifier "MWo"
    /* 0x3 */ uint8_t version;         // MWo version
    /* 0x4 */ uint32_t overlayID;      // Overlay ID
    /* 0x8 */ uint32_t address;        // Load address (ie. address of this structure, followed by the data)  
    /* 0xC */ uint32_t textSize;       // Size of the .text section   
    /* 0x10 */ uint32_t dataSize;      // Size of the .data section
    /* 0x14 */ uint32_t bssSize;       // Size of the .bss section
    /* 0x18 */ uint32_t staticInit;    // Start address of the static array of initialization function pointers
    /* 0x1C */ uint32_t staticInitEnd; // End address of the static array of initialization function pointers
    /* 0x20 */ uint8_t name[32];       // Overlay name
    } mwOverlayHeader; // size: 0x40/64 bytes
    """

    def __init__(self, obj: Union[Path, str, bytes]) -> None:
        if isinstance(obj, Path) or isinstance(obj, str):
            file_path = Path(obj).resolve()
            if file_path and file_path.exists():
                self.extract_header(file_path.read_bytes())
            else:
                raise FileNotFoundError(f"{file_path} does not exist or is invalid")
        else:
            self.extract_header(obj)

    def __repr__(self) -> str:
        return (
            f"MwOverlayHeader(identifier='{self.identifier}', mwo_version={self.mwo_version}, "
            f"overlay_id={self.overlay_id}, address={self.address}, text_size={self.text_size}, "
            f"data_size={self.data_size}, bss_size={self.bss_size}, static_init_start=0x{self.static_init_start}, "
            f"static_init_end={self.static_init_end}, name='{self.name}')"
        )

    def extract_header(
        self, data: bytes, struct_format: str = "<3sBIIIIIII32s"
    ) -> MwOverlayHeader:
        if not isinstance(struct_format, str):
            raise TypeError(f"Format must be a string, but got {type(struct_format)}")
        if not isinstance(data, bytes):
            raise TypeError(
                f"Data must be provided as {type(bytes())}, {type(Path())}, or {type(str())} but got {type(data)}"
            )
        format_size = struct.calcsize(struct_format)
        if len(data) < format_size:
            raise ValueError(
                f"Data size must be >= {format_size} bytes, but got {len(data)}"
            )
        unpacked_data = struct.unpack(struct_format, data[:format_size])
        self.identifier = unpacked_data[0].decode("ascii")
        self.mwo_version = unpacked_data[1]
        self.overlay_id = unpacked_data[2]
        self.address = yaml.Hex(unpacked_data[3])
        self.text_size = yaml.Hex(unpacked_data[4])
        self.data_size = yaml.Hex(unpacked_data[5])
        self.bss_size = yaml.Hex(unpacked_data[6])
        self.static_init_start = yaml.Hex(unpacked_data[7])
        self.static_init_end = yaml.Hex(unpacked_data[8])
        self.name = unpacked_data[9].rstrip(b"\x00").decode("ascii")

        return self


def get_text_offset(data: bytes) -> Optional[int]:
    addiu_sp = mips.Instruction.from_fields(
        opcode=mips.Opcode.ADDIU.value,
        rs=mips.Register.SP.value,
        rt=mips.Register.SP.value,
    ).instruction.lstrip(b"\x00")
    # Search for 'addiu $sp, $sp, imm address'
    text_offset = data.find(addiu_sp) - 2

    # Checks each addiu $sp, $sp match until it finds one that is both
    # in the proper byte alignment and the imm address is not 0
    while text_offset > 0 and (
        text_offset % 4 != 0
        or (data[text_offset + 1] != b"\x00" and data[text_offset + 1] != 0xFF)
        or data[text_offset : text_offset + 2] == b"\x00" * 2
    ):
        text_offset = data.find(addiu_sp, text_offset + 4) - 2

    return text_offset if text_offset > 0 else None


def get_bss_offset(data: bytes) -> Optional[int]:
    # Find the final 'jr $ra' to identify the likely end of the text section
    bss_offset = data.rfind(
        mips.Instruction.from_fields(
            funct=mips.Opcode.JR.value, rs=mips.Register.RA.value
        ).instruction
    )
    return None if bss_offset == -1 else bss_offset + 8


def get_rodata_address(data: bytes) -> Optional[int]:
    # Look for 'jr $v0'
    jr_offset = data.find(
        mips.Instruction.from_fields(
            funct=mips.Opcode.JR.value, rs=mips.Register.V0.value
        ).instruction
    )
    if jr_offset == -1:
        return None

    lw_v0 = mips.Instruction.from_fields(
        opcode=mips.Opcode.LW.value, rt=mips.Register.V0.value
    )
    # Look for last 'lw $v0, %lo(XXX)(YYY)' before jr_offset
    lw_offset = data.rfind(lw_v0.instruction[3], 0, jr_offset) - 3
    if lw_offset == -1:
        return None

    lw_v0 = mips.Instruction.from_bytes(data[lw_offset : lw_offset + 4])
    lui_rs = mips.Instruction.from_fields(
        opcode=mips.Opcode.LUI.value, rt=lw_v0.rs, rs=0
    ).instruction
    # Look for last 'lui $at, %hi(XXX) before lw_offset
    lui_offset = data.rfind(lui_rs.lstrip(b"\x00"), 0, jr_offset) - 2

    if lui_offset == -1:
        return None

    return (
        mips.Instruction.from_bytes(data[lui_offset : lui_offset + 4]).immu << 16
    ) + lw_v0.imm


def find_segments(ovl_config, file_header, known_segments):
    logger = get_logger()
    segments = []
    rodata_pattern = re.compile(
        rf"glabel (?:jtbl|D)_{ovl_config.version}"
        + r"_[0-9A-F]{8}\n\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s"
    )
    camel_case_pattern = re.compile(r"([A-Za-z])([A-Z][a-z])")
    include_rodata_pattern = re.compile(
        r'INCLUDE_RODATA\("[A-Za-z0-9/_]+",\s?(?P<name>\w+)\);'
    )
    include_asm_pattern = re.compile(
        r'INCLUDE_ASM\("(?P<dir>[A-Za-z0-9/_]+)",\s?(?P<name>\w+)\);'
    )

    src_text = ovl_config.first_src_file.read_text()

    segment_meta = None
    functions = deque()
    matches = include_asm_pattern.findall(src_text)
    for i, match in enumerate(matches):
        asm_dir, current_function = match
        current_function_parts = current_function.split("_")
        if current_function.startswith(f"func_{ovl_config.version}_"):
            current_function_stem = "_".join(current_function_parts[:3])
        elif current_function_parts[0] == "func":
            current_function_stem = "_".join(current_function_parts[:2])
        elif current_function_parts[0] == "GetLang":
            current_function_stem = current_function_parts[0]
        else:
            current_function_stem = current_function

        in_known_segment = bool(
            segment_meta
            and (
                segment_meta.end
                or (segment_meta.allow and current_function_stem in segment_meta.allow)
            )
        )

        if (
            current_function_parts[0] == "GetLang"
            and matches[i + 1][1] in known_segments
        ) or (
            current_function_parts[0] != "GetLang"
            and current_function_stem in known_segments
            and not in_known_segment
            and (
                not segment_meta
                or not segment_meta.name
                or not segment_meta.name.endswith(
                    known_segments[current_function_stem].name
                )
            )
        ):
            if segment_meta:
                if not segment_meta.name and len(functions) == 1:
                    segment_meta.name = f"{ovl_config.segment_prefix}{camel_case_pattern.sub(r'\1_\2', functions[0]).lower().replace('entity', 'e')}"
                if not functions:
                    logger.error(
                        f"Found start function {current_function} that isn't allowed for {segment_meta.name}, this is likely an error in segments.yaml"
                    )
                segment_meta.end = functions[-1]
                logger.debug(
                    f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
                )
                segments.append(segment_meta)
                functions.clear()
                segment_meta = None

            if current_function_parts[0] == "GetLang":
                segment_meta = known_segments[matches[i + 1][1]]
                segment_meta.start = current_function
            elif current_function_stem not in known_segments:
                for num in range(len(current_function_parts), 0, -1):
                    if "_".join(current_function_parts[:num]) in known_segments:
                        segment_meta = known_segments[
                            "_".join(current_function_parts[:num])
                        ]
                        segment_meta.start = current_function
            else:
                segment_meta = known_segments[current_function_stem]
                segment_meta.start = current_function
            segment_meta.offset = None
            if ovl_config.version == "pspeu":
                segment_meta.name = f"{ovl_config.segment_prefix}{segment_meta.name}"
            segment_meta.asm_dir = asm_dir
        elif not segment_meta:
            segment_meta = SimpleNamespace(
                name=None,
                start=current_function,
                end=None,
                asm_dir=asm_dir,
                offset=None,
                allow=None,
            )

        if segment_meta and not segment_meta.offset:
            address = get_symbol_address(
                ovl_config.ld_script_path.with_suffix(".map"), current_function
            )
            if address:
                offset = address - ovl_config.vram + ovl_config.start
                segment_meta.offset = SimpleNamespace(int=offset)
                segment_meta.offset.str = f"{segment_meta.offset.int:X}"
            else:
                asm_path = (
                    Path("asm") / ovl_config.version / asm_dir / f"{current_function}.s"
                )
                asm_text = asm_path.read_text()
                if first_offset := re.search(
                    rf"glabel {current_function}"
                    + r"\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s",
                    asm_text,
                ):
                    segment_meta.offset = SimpleNamespace(str=first_offset.group(1))
                    segment_meta.offset.int = int(segment_meta.offset.str, 16)
        if not segment_meta.name and segment_meta.offset:
            segment_meta.name = (
                f"{ovl_config.segment_prefix}unk_{segment_meta.offset.str}"
            )

        if segment_meta and current_function_stem == segment_meta.end:
            logger.debug(
                f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
            )
            segments.append(segment_meta)
            functions.clear()
            segment_meta = None
        elif (
            segment_meta
            and segment_meta.allow
            and current_function_stem not in segment_meta.allow
        ):
            logger.debug(
                f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
            )
            if not functions:
                logger.error(
                    f"Found start function {current_function} that isn't allowed for {segment_meta.name}, this is likely an error in segments.yaml"
                )
            segment_meta.end = functions[-1]
            segments.append(segment_meta)
            functions.clear()
            segment_meta = SimpleNamespace(
                name=None,
                start=current_function,
                end=None,
                asm_dir=asm_dir,
                offset=None,
                allow=None,
            )
            address = get_symbol_address(
                ovl_config.ld_script_path.with_suffix(".map"), current_function
            )
            if address:
                offset = address - ovl_config.vram + ovl_config.start
                segment_meta.offset = SimpleNamespace(int=offset)
                segment_meta.offset.str = f"{segment_meta.offset.int:X}"
            else:
                asm_path = (
                    Path("asm") / ovl_config.version / asm_dir / f"{current_function}.s"
                )
                asm_text = asm_path.read_text()
                if first_offset := re.search(
                    rf"glabel {current_function}"
                    + r"\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s",
                    asm_text,
                ):
                    segment_meta.offset = SimpleNamespace(str=first_offset.group(1))
                    segment_meta.offset.int = int(segment_meta.offset.str, 16)
            functions.append(current_function)
        else:
            functions.append(current_function)

    if segment_meta and segment_meta not in segments:
        # Todo: Handle this without duplicating the code from the loop, if possible
        if not segment_meta.name and len(functions) == 1:
            # Todo: Only change name if it isn't a defined segment
            segment_meta.name = f'{ovl_config.segment_prefix}{camel_case_pattern.sub(r"\1_\2", functions[0]).lower().replace("entity", "e")}'
        logger.debug(
            f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
        )
        segments.append(segment_meta)

    segments = tuple(segments)

    rodata_subsegments = [
        subseg
        for subseg in ovl_config.subsegments
        if len(subseg) >= 2
        and "rodata" in subseg[1]
        and ovl_config.first_src_file.stem not in subseg[2]
    ]

    for segment in segments:
        # Todo: Decide if these need to be more flexible about the existence of the leading space
        first_function_index = src_text.find(f" {segment.start});")
        last_function_index = src_text.find(f" {segment.end});")
        segment_start = src_text[:first_function_index].rfind("INCLUDE_ASM")
        if (segment_end := src_text[last_function_index:].find("INCLUDE_ASM")) == -1:
            segment_end = len(src_text)
        else:
            segment_end += last_function_index
        segment_text = (
            src_text[segment_start:segment_end]
            .replace(
                f"{ovl_config.nonmatchings_path}/{ovl_config.segment_prefix}{ovl_config.first_src_file.stem}",
                f"{ovl_config.nonmatchings_path}/{segment.name}",
            )
            .rstrip("\n")
        )

        # Extract rodata symbols from INCLUDE_RODATA macros
        for rodata_symbol in include_rodata_pattern.findall(segment_text):
            rodata_address = get_symbol_address(
                ovl_config.ld_script_path.with_suffix(".map"), rodata_symbol
            )
            if rodata_address:
                rodata_offset = rodata_address - ovl_config.vram + ovl_config.start
                rodata_subsegments.append(
                    SimpleNamespace(
                        offset=rodata_offset, type=".rodata", name=segment.name
                    )
                )

        # Extract rodata offsets from assembly files referenced in INCLUDE_ASM macros
        asm_files = [
            ovl_config.asm_path.joinpath(
                ovl_config.nonmatchings_path,
                ovl_config.segment_prefix,
                ovl_config.first_src_file.stem,
                match.group(2),
            ).with_suffix(".s")
            for match in include_asm_pattern.finditer(segment_text)
        ]

        for asm_file in asm_files:
            asm_text = asm_file.read_text()
            rodata_start = asm_text.find(".section .rodata")
            text_start = asm_text.find(".section .text")
            if rodata_start != -1 and (text_start > rodata_start or text_start == -1):
                rodata_text = (
                    asm_text[rodata_start:text_start]
                    if text_start > rodata_start
                    else asm_text[rodata_start:]
                )
                for rodata_offset in rodata_pattern.findall(rodata_text):
                    rodata_subsegments.append(
                        SimpleNamespace(
                            offset=int(rodata_offset, 16),
                            type=".rodata",
                            name=segment.name,
                        )
                    )

        ovl_config.src_path.joinpath(segment.name).with_suffix(".c").write_text(
            file_header + segment_text + "\n"
        )

    rodata_by_segment = {}
    for rodata_subsegment in rodata_subsegments:
        if (
            rodata_subsegment.name not in rodata_by_segment
            or rodata_subsegment.offset < rodata_by_segment[rodata_subsegment.name][0]
        ):
            rodata_by_segment[rodata_subsegment.name] = [
                rodata_subsegment.offset,
                rodata_subsegment.type,
                rodata_subsegment.name,
            ]

    ovl_config.first_src_file.unlink()

    # Todo: Add ability to postprocess selective segments into comments.  bss segments can't be split until their files are fully imported.
    """first_bss_index = next(i for i,subseg in enumerate(ovl_config.subsegments) if "bss" in subseg or "sbss" in subseg)
    bss_subsegs = [ovl_config.subsegments[first_bss_index]] if ovl_config.subsegments[first_bss_index][0] != create_entity_bss_start else []
    bss_subsegs.extend([yaml.FlowSegment([create_entity_bss_start, ".bss" if ovl_config.platform == "psp" else ".sbss", f"{ovl_config.name}_psp/create_entity" if ovl_config.version == "pspeu" else "create_entity"]), yaml.FlowSegment([create_entity_bss_end, "bss"])])
    ovl_config.subsegments[first_bss_index:first_bss_index+1] = bss_subsegs"""

    first_text_index = next(
        i for i, subseg in enumerate(ovl_config.subsegments) if "c" in subseg
    )
    text_subsegs = [
        yaml.FlowSegment([segment.offset.int, "c", segment.name])
        for segment in segments
    ]
    rodata_subsegs = [yaml.FlowSegment(x) for x in tuple(rodata_by_segment.values())]
    ovl_config.subsegments[first_text_index : first_text_index + 1] = text_subsegs
    first_rodata_index = next(
        (i for i, subseg in enumerate(ovl_config.subsegments) if ".rodata" in subseg),
        None,
    )
    if first_rodata_index:
        ovl_config.subsegments[first_rodata_index : first_rodata_index + 1] = (
            rodata_subsegs
        )

    return sort_subsegments(ovl_config.subsegments)
