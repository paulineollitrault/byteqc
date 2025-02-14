# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# ByteQC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ByteQC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os

pwd = os.path.abspath(__file__)[:-9]
lib = build = os.path.join(pwd, 'lib')
build = os.path.join(lib, 'build')
print("Begin to build subpackage cupbc")
if not os.path.exists(build):
    os.system('mkdir %s' % build)
if os.system('cmake %s -B %s' % (lib, build)) == 0:
    if os.system('make -j -C %s' % build) == 0:
        print("\033[42mBuilt done for cupbc!\033[0m\n\n")
        exit()
print("\033[41mBuilt failed for cupbc!\033[0m\n\n")
exit(1)
