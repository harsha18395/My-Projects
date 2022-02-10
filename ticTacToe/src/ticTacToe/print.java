package ticTacToe;

public class print {
	public static void showGame(char[][] values) {
		System.out.println("  A   B   C");
		for(int i =0;i<3;i++)
		{
			System.out.print(i+1);
			for(int j =0;j<3;j++)
			{
				char c = '_';
				if(values[i][j]!='\0')
					c = values[i][j]; // put '_' if there is no value
				System.out.print(" "+c+" |");
			}
			System.out.println();
		}
	}
}
