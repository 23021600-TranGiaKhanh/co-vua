#include <iostream>

#include "position.h"
#include "bitboards.h"
#include "attacks.h"
#include "uci.h"
#include "transpositiontable.h"
#include "pawnhashtable.h"

using namespace std;

int main() {

	// Initialization
	Position::init();
	Bitboards::init();
	Attacks::init();
	Search::init();
	Evaluation::init();

	// LICENSE
	cout << "**************************************************************" << endl;
	cout << "* MORA CHESS ENGINE (MCE) - Copyright (C) 2019 Gonzalo Arró. *" << endl;
	cout << "**************************************************************" << endl;
	cout << "This program comes with ABSOLUTELY NO WARRANTY." << endl;;
	cout << "This is free software, and you are welcome to redistribute it under certain conditions." << endl;
	cout << "See COPYING file for details." << endl << endl;

	// UCI Protocol
	UCI::loop();

	return 0;
}
