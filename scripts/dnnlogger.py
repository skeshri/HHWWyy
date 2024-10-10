# -*- coding: utf-8 -*-
# @Author: Ram Krishna Sharma
# @Date:   2021-04-05 20:57:39
# @Last Modified by:   Ram Krishna Sharma
# @Last Modified time: 2021-04-05 20:58:06
class colors:
    class foreground:
        # Foreground
        WHITE     = '\033[37m'
        CYAN      = '\033[36m'
        MAGENTA   = '\033[35m'
        BLUE      = '\033[34m'
        YELLOW    = '\033[33m'
        GREEN     = '\033[32m'
        RED       = '\033[31m'
        BLACK     = '\033[30m'
        END       = '\033[39m'
        pass

    class background:
        # Background
        WHITE     = '\033[47m'
        CYAN      = '\033[46m'
        MAGENTA   = '\033[45m'
        BLUE      = '\033[44m'
        YELLOW    = '\033[43m'
        GREEN     = '\033[42m'
        RED       = '\033[41m'
        BLACK     = '\033[40m'
        END       = '\033[49m'
        pass

    WHITE     = '\033[97m'
    CYAN      = '\033[96m'
    MAGENTA   = '\033[95m'
    BLUE      = '\033[94m'
    YELLOW    = '\033[93m'
    GREEN     = '\033[92m'
    RED       = '\033[91m'
    BLACK     = '\033[90m'
    END       = '\033[99m'

    # styles
    class style:
        DIM       = '\033[2m'
        BRIGHT    = '\033[1m'
        NORMAL    = '\033[22m'
        UNDERLINE = '\033[4m'
        pass

    # resets
    ENDC      = '\033[0m'

    # log levels
    colorlevels = {
        "FATAL"   :"%s%s%s"%(background.RED,style.BRIGHT,YELLOW),
        "CRITICAL":"%s%s%s"%(background.RED,style.BRIGHT,YELLOW),
        "ERROR"   :"%s%s"%(style.BRIGHT,RED),
        "WARNING" :"%s%s"%(style.BRIGHT,YELLOW),
        "WARN"    :"%s%s"%(style.BRIGHT,YELLOW),
        "INFO"    :"%s%s"%(style.BRIGHT,GREEN),
        "DEBUG"   :"%s%s"%(style.BRIGHT,MAGENTA),
        "NOTSET"  :ENDC
        }

    pass
