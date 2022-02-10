package ticTacToe;
import java.util.Scanner;
public class ApplicationTicTacToe {
	public static void main(String[] args) {
			
		Scanner obj = new Scanner(System.in);
		while(true) 
		{
			System.out.println("Shall we play a game?");
			String dec =  obj.nextLine();
			if(dec.equalsIgnoreCase("no"))
			{
				System.out.println("Have a good day. Bye");
				break; //End Game
			}
			
			int count = 0;
			store gameManager= store.getInstance();
			gameManager.printValues();
			while(count<9)
			{	
				System.out.println();
				System.out.println("Player, make your move !");
				boolean playerPlayed =false;
				while(!playerPlayed) // ask player to play until he makes correct move
				{
					String pos = obj.nextLine();
					try {
						gameManager.fillXOs(pos, true);
					}
					catch(IllegalArgumentException e) {
						System.err.println(e.getMessage());
						continue; // ask user to enter again
					}
				
					playerPlayed = true;
					count++;
				}
				
				if(checkWin.didUserWin()) // check if game ended
				{
					gameManager.printValues();
					System.out.println("Congratulations, you win!");
					break; // Start the game again
				}
				
				if(count<9)
				{
				  gameManager.playHardOpp();
				  count++;
				  gameManager.printValues();
				  if(checkWin.didOppWin()) // check if game ended
					{
						System.out.println("Sorry, you Lose!");
						break; // Start the game again
					}
				}
				else
					break;// Start the game again
			}
			
			if(count >=9 && (!checkWin.didUserWin() && !checkWin.didOppWin()))
			{
			gameManager.printValues();
			System.out.println();
			System.out.println("A STRANGE GAME.\nTHE ONLY WINNING MOVE IS NOT TO PLAY.");
			}
			
			gameManager.refresh();
			checkWin.reset();
			
		} // Ask user to start new game 
			
			obj.close();
			return;
	}
}
