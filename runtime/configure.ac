## Process this file with autoconf 2.69. Other versions may not work.
AC_INIT([gemmini-driver],[1.0])

AC_PROG_CC
AC_PROG_FGREP
AC_PROG_GREP

AC_PATH_PROG([BASH], [bash],
	[AC_MSG_ERROR([bash not found])])

AC_ARG_ENABLE(debug_info,
        [AS_HELP_STRING([--enable-debug-info],
		[build gemmini driver with debug information])],
        [enable_debug_info=yes],
        [enable_debug_info=no]
        )

AS_IF([test "x$enable_debug_info" != xyes],
        [disable_debug_info=yes],
        [disable_debug_info=no]
        )

AS_IF([test "x$enable_debug_info" != xyes],
	[AC_SUBST(debug_info, "")],
	[AC_SUBST(debug_info, "-g")])

AC_DEFUN([AX_ARG_WITH_SRC],
        [{m4_pushdef([opt_name], with_$1)
          AC_ARG_WITH($1,
                [AS_HELP_STRING([--with-$1],[Set $1 path])],
                [],
                [opt_name=default]
                )
          AS_IF([test "x$opt_name" != xdefault],
                [AC_SUBST(opt_name,$opt_name)],
                [AC_SUBST(opt_name,"\$(srcdir)/$2")])
          m4_popdef([opt_name])
        }])

AX_ARG_WITH_SRC(gemmini_driver, gemmini_driver)

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
