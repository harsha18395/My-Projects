package ticTacToe;

import java.util.Random;

public class HardOpp {
	
	private static char[][]ivalues = new char[3][3];
	private static boolean lDiag= false;
	private static boolean rDiag= false;
	private static int[] pos= new int[2];
	
	private static int checkRowCount(int row,char c) 
	{
		int count =0;
		for(int i=0;i<3;i++) 
		{
			if(ivalues[row][i]==c)
				count++;
		}
		
		return count;
	}
	
	private static int checkColCount(int col,char c) 
	{
		int count =0;
		for(int i=0;i<3;i++)
		{
			if(ivalues[i][col]==c)
				count++;
		}
		
		return count;
	}
	
	private static boolean isDiagonal(int row,int col)
	{
		boolean isDiag = false;
		if((row == 0||row==2)  && (col == 0 || col ==2))
		{
			isDiag = true;
			if((row==0 && col==0) || (row==2 && col==2) )
				lDiag =true;
			else if((row==0 && col==2) || (row==2 && col==0) )
				rDiag =true;
		}
			
		else if (row == 1 && col == 1)
		{
			isDiag = true;
			lDiag = true;
			rDiag = true;
		}
		
		return isDiag;
	}
	
	private static int checkLDiagCount(char c) 
	{
		int lcount =0;

		for(int i=0,j=0;i<3;i++,j++)
		{
				if(ivalues[i][j]== c)
					lcount++;
		}
		return lcount; 
	}
	
	private static int checkRDiagCount(char c) 
	{
		int rcount =0;

		for(int i=0,j=2;i<3;i++,j--)
		{
				if(ivalues[i][j]==c)
					rcount++;
		}
		
		return rcount;
	}
	
	private static void getPos(String type,int row,int col)
	{
		
		
		switch(type)
		{
		case "row":
			for(int i=0;i<3;i++) 
			{
				if(ivalues[row][i]=='\0')
				{
					pos[0] = row ;
					pos[1] = i;
					break;
				}
			}
			break;
		
		case "col":
			for(int i=0;i<3;i++) 
			{
				if(ivalues[i][col]=='\0')
				{
					pos[0] = i ;
					pos[1] = col;
					break;
				}
			}
			break;
			
		case "lDiag":
			for(int i=0,j=0;i<3;i++,j++)
			{
				if(ivalues[i][j]=='\0')
				{
					pos[0] = i ;
					pos[1] = j;
					break;
				}
			}
			break;
			
		case "rDiag":
			for(int i=0,j=2;i<3;i++,j--)
			{
				if(ivalues[i][j]=='\0')
				{
					pos[0] = i ;
					pos[1] = j;
					break;
				}
			}
			break;
		}
		
		
		  if(ivalues[pos[0]][pos[1]]!='\0')
		  { 
			  Random rand = new Random(); 
			  int r1=0,r2=0; boolean alreadyFilled = true; 
			  while(alreadyFilled) {
				  r1 =rand.nextInt(3)+0; 
				  r2 = rand.nextInt(3)+0; 
				  if(ivalues[r1][r2]== '\0')
					  alreadyFilled = false;
				  }
		  
			  pos[0] = r1; pos[1] = r2;
		  }
		 
	}
	
	private static boolean isPossibilityToWin() 
	{
		for(int i =0 ; i<3;i++)
		{
			if(checkRowCount(i, 'O')==2)
			{
				if(checkRowCount(i, '\0')==1)
				{
					getPos("row",i,0);
					return true;
				}
			}
			
			if(checkColCount(i, 'O')==2)
			{
				if(checkColCount(i, '\0')==1)
				{
					getPos("col",0,i);
					return true;
				}	
			}
		}	
			if(checkLDiagCount('O')==2)
			{
				if(checkLDiagCount('\0')==1)
				{
					getPos("lDiag",0,0);
					return true;
				}	
			}
			
			if(checkRDiagCount('O')==2)
			{
				if(checkRDiagCount('\0')==1)
				{
					getPos("rDiag",0,0);
					return true;
				}	
			}
		
		return false;
	}
	
	private static void getBestPosition() {
		int row = store.getLastUserMove()[0];
		int col = store.getLastUserMove()[1];
		int rCount = checkRowCount(row,'X');
		int cCount = checkColCount(col,'X');
		
		if(isPossibilityToWin())
			return;
		
		lDiag = false;
		rDiag = false;
		String type = "";
		if(isDiagonal(row, col))
		{
			if(rCount ==2 || cCount ==2 )
				type = (rCount > cCount) ? "row":"col";
			else
			{
			 if(lDiag && rDiag)
			 {
				int rDCount = checkRDiagCount('X');
				int lDCount = checkLDiagCount('X');
				
				if(lDCount > rDCount)
					rDiag = false;
				else
					lDiag = false;
			 }
			 if(lDiag)
			  type = "lDiag";
			 else
			  type = "rDiag";			
			}
		}
		else 
		{
			type = (rCount > cCount) ? "row":"col";
			
		}
		getPos(type, row, col);		
	}
	
	// method called to get best position
	public static int[] playHardOpponent(char[][] values) {
		ivalues = values;
		//System.out.println(ivalues[0][0] );
		getBestPosition();
		return pos;
	}
}
