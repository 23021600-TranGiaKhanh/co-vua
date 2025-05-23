#ifndef SRC_MOVEGENERATOR_H_
#define SRC_MOVEGENERATOR_H_

#include "position.h"

namespace MoveGen {

	/*
	 * There are legal positions with more moves than 80
	 * but those are unlikely to be reached, and is more efficient
	 * to have a smaller array in the Move_list structure.
	 */
	constexpr int MAX_POSSIBLE_MOVES = 80;

	/*
	 * Move list struct to save generated moves.
	 */
	struct Move_list {
		Move moves[MAX_POSSIBLE_MOVES];
		int size;
		Move_list() : size(0) {};
	};

	/*
	 * Generate pseudo-legal moves in the position.
	 */
	void generate_moves(Position &pos, Move_list &move_list);

	/*
	 * Generate pseudo-legal captures in the position.
	 */
	void generate_captures(Position &pos, Move_list &move_list);

	/*
	 * Generate pseudo-legal promotions in the position.
	 */
	void generate_promotions(Position &pos, Move_list &move_list);
}

#endif /* SRC_MOVEGENERATOR_H_ */
