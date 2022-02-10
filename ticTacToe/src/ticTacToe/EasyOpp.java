package ticTacToe;

import java.util.Random;

public class EasyOpp {
	public static int[] playEasyOpponent(char[][] values) {
		Random rand = new Random();
		int r1=0,r2=0;
		boolean alreadyFilled = true;
		while(alreadyFilled) 
		{
		r1 = rand.nextInt(3)+0;
		r2 = rand.nextInt(3)+0;
		if(values[r1][r2]== '\0')
			alreadyFilled = false;
		}
		
		int[] pos = {r1,r2};
		return pos;
	}
}
