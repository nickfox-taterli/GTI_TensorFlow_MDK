'''
All Rights Reserved.

Copyright (c) 2017-2019, Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
"""Configurae and generate chip .DAT file."""
from ctypes import CDLL, c_char_p, c_bool, c_ulonglong
import json
import os
import platform


if platform.system() != "Linux":
    raise NotImplementedError("Windows support is currently limited")


def gticonfig(dat_json, filter_file, bias_file, dat_out, save_dir, debug=False):
    tb = os.path.join(save_dir, "gti.tb")
    with open(dat_json, "r") as f:
        chip_type = str(json.load(f)["model"][0]["ChipType"])
        lib = CDLL(os.path.join(os.path.dirname(__file__), "libgticonfig" + chip_type + ".so"))
        lib.GtiConvertInternalToSDK.argtypes = (
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_bool
        )
        lib.GtiConvertInternalToSDK(
            dat_json.encode("ascii"),
            filter_file.encode("ascii"),
            bias_file.encode("ascii"),
            ("GTI" + chip_type).encode("ascii"),
            dat_out.encode("ascii"),
            tb.encode("ascii"),
            save_dir.encode("ascii"),
            debug 
        )