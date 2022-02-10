package ticTacToe;

import java.util.HashMap;
import java.util.Map;

public class store {
	private static store storeInstance;
	private Map<String,Integer> position ;
	private char[][] x_os;
	private static int[] lastUserMove = new int[2];
	
	//private constructor for singleton
	private store()
	{
		x_os = new char[3][3];
		position = new HashMap<String,Integer>();
		position.put("A",0);
		position.put("B",1);
		position.put("C",2);
	}
	
	// get instance 
	public static store getInstance() 
	{
		if(storeInstance == null)
		{
			storeInstance = new store();
		}
		
		return storeInstance;
	}
	
	// to get last user move
	public static int[] getLastUserMove() {
		return lastUserMove;
	}
	
	// fill X and O's given position in terms of 'A1' , 'B2' ..etc
	public void fillXOs(String pos,boolean user) 
	{
		String[] str = pos.split("");
		
		if(str.length !=2 ||  !position.containsKey(str[0].toUpperCase()) ) 
		{
			throw new IllegalArgumentException("Please enter valid position");
		}
			
		int p1 = Integer.parseInt(str[1]);
		if( p1 <0 || p1>3) 
		{
			throw new IllegalArgumentException("Please enter valid position");
		}
		
		int p2 =position.get(str[0].toUpperCase());
		
		fillXOs_ext(p1-1,p2,user);
		lastUserMove[0] = p1-1;
		lastUserMove[1]= p2;
	}
	
	private void fillXOs_ext(int row,int col,boolean user) 
	{

		if(x_os[row][col]!= '\0' ) 
		{
			throw new IllegalArgumentException("Position already filled .Please enter valid position");
		}
		char inp ='O';
		if(user)
			inp = 'X'; //if User inputs put 'X' else put 'O'
		x_os[row][col] = inp;
		checkWin.checkGame(row, col, x_os, user);
		
	}
	// print UI
	public void printValues()
	{
		print.showGame(x_os);
	}
	
	public void playEasyOpp() 
	{
		int[] pos = EasyOpp.playEasyOpponent(x_os);
		fillXOs_ext(pos[0],pos[1],false);
	}
	
	public void playHardOpp() 
	{
		int[] pos = HardOpp.playHardOpponent(x_os);
		//System.out.println(pos[0] + "" +pos[1]);
		fillXOs_ext(pos[0],pos[1],false);
	}
	
	// refresh the array to start from beginning
	public void refresh() 
	{
		for(int i =0;i<3;i++) 
		{
			for(int j =0;j<3;j++) 
			{
				x_os[i][j] = '\0';     // refreshing the array for new game
			}
		}
	}
}
