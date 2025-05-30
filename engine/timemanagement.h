#ifndef SRC_TIMEMANAGEMENT_H_
#define SRC_TIMEMANAGEMENT_H_

#include "types.h"

namespace Time {

	// Constants
	constexpr int max_time_to_search = 3600000; // one hour

	struct Time_options {
		int time_left;
		int moves_to_go;
		bool infinite;
	};

	/*
	 * Assigns a certain amount of time for the search
	 * of the next move.
	 * Returns the time in milliseconds.
	 */
	int get_time_to_search(Time_options &options, int moves_so_far);

	/*
	 * Returns the current time of the system
	 * in milliseconds.
	 */
	long long get_current_time_in_milliseconds();

	/*
	 * Indicates a time out event when the time
	 * assigned for the search has finished.
	 */
	bool time_out(long long start_time, int time_to_search);

	/*
	 * Decides whether there is time for another iteration
	 * inside the iterative deepening framework of the search
	 * algorithm.
	 */
	bool time_for_next_iteration(long long start_time, int time_to_search);
}

#endif /* SRC_TIMEMANAGEMENT_H_ */
