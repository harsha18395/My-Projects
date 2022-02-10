package ticTacToe;

public class checkWin {
	private static boolean userWin = false;
	private static boolean OpponentWin = false;
	private static char[][]values = new char[3][3];
	
	// to reset results
	public static void reset() {
		userWin = false;
		OpponentWin = false;
	}
	
	public static boolean didUserWin() {
		return userWin;
	}
	
	public static boolean didOppWin() {
		return OpponentWin;
	}
	// to check if the value is part of diagonal
	private static boolean isDiagonal(int row,int col)
	{
		boolean isDiag = false;
		if((row == 0||row==2)  && (col == 0 || col ==2))
		{
			isDiag = true;
		}
			
		else if (row == 1 && col == 1)
			isDiag = true;
		
		return isDiag;
	}
	
	// to check if particual column gets the winner
	private static boolean checkCol(int col)
	{
		if(values[0][col]==values[1][col] && values[2][col]==values[1][col] && values[0][col]!='\0')
			return true;
		
		return false;
	}
	
	// to check if particual row gets the winner
	private static boolean checkRow(int row)
	{
		if(values[row][0]==values[row][1] && values[row][2]==values[row][1] && values[row][0]!='\0')
			return true;
		
		return false;
	}
	
	// to check if diagonal gets the winner based on the character X or O
	private static boolean checkDiagonal(char c)
	{
		if(values[0][0]==values[1][1] && values[2][2]==values[1][1] && values[0][0]==c)
			return true;
		else if (values[2][0]==values[1][1] && values[0][2]==values[1][1] && values[2][0]==c)
			return true;
		
		return false;
	}
	
	// main method to be called to check the winner
	public static void checkGame(int row,int col,char[][]ivalues,boolean user) {
		values = ivalues;
		char c = 'O';
		if(user)
			c = 'X';
		
		boolean isItWin = false;
		
		if(checkRow(row) || checkCol(col))
		{
			isItWin = true;
			if(user)
				userWin = true;
			else
				OpponentWin = true;
			return;
		}
		
		if(isDiagonal(row,col) && checkDiagonal(c))//check diagonal 
		{
			if(user)
				userWin = true;
			else
				OpponentWin = true;
			return;
		}
	}
}
