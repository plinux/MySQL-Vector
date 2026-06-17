/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is designed to work with certain software (including
   but not limited to OpenSSL) that is licensed under separate terms,
   as designated in a particular file or component or in included license
   documentation.  The authors of MySQL hereby grant you an additional
   permission to link the program and your derivative works with the
   separately licensed software that they have either included with
   the program or referenced in the documentation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include <pthread.h>
#include <signal.h>

namespace {

/**
  Block server control signals before dynamic-library initializers run.

  Some vector dependencies create worker threads before main(). Those threads
  must inherit the normal server signal mask before my_init_signals() later
  installs the final main-thread mask, so the MySQL signal thread remains the
  only consumer of server control signals.
*/
void block_server_control_signals() {
  sigset_t signal_mask;
  (void)sigemptyset(&signal_mask);
  (void)sigaddset(&signal_mask, SIGINT);
  (void)sigaddset(&signal_mask, SIGQUIT);
  (void)sigaddset(&signal_mask, SIGHUP);
  (void)sigaddset(&signal_mask, SIGTERM);
  (void)sigaddset(&signal_mask, SIGTSTP);
  (void)sigaddset(&signal_mask, SIGUSR1);
  (void)sigaddset(&signal_mask, SIGUSR2);
  (void)pthread_sigmask(SIG_BLOCK, &signal_mask, nullptr);
}

using preinit_function = void (*)();

__attribute__((section(".preinit_array"), used))
preinit_function vector_dependency_signal_preinit =
    block_server_control_signals;

}  // namespace
