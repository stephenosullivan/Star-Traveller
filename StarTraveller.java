import java.io.*;

public class StarTraveller {
    int NStars;
    boolean[] used;
    public int init(int[] stars) {
        NStars = stars.length / 2;
        used = new boolean[NStars];
        return 0;
    }
    public int[] makeMoves(int[] ufos, int[] ships) {
        int[] ret = new int[ships.length];
        int retInd = 0;
        for (int i = 0; i < NStars; ++i)
            if (!used[i])
            {
                used[i] = true;
                ret[retInd] = i;
                ++retInd;
                if (retInd == ships.length)
                    break;
            }
        while (retInd < ships.length)
        {
            ret[retInd] = (ships[retInd] + 1) % NStars;
            ++retInd;
        }
        return ret;
    }
    // -------8<------- end of solution submitted to the website -------8<-------
    public static void main(String[] args) {
      try {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        
        int NStars = Integer.parseInt(br.readLine());
        int[] stars = new int[NStars];
        for (int i = 0; i < NStars; ++i) {
            stars[i] = Integer.parseInt(br.readLine());
        }
        
        StarTraveller algo = new StarTraveller();
        int ignore = algo.init(stars);
        System.out.println(ignore);
        System.out.flush();
        
        while (true)
        {
            int NUfo = Integer.parseInt(br.readLine());
            if (NUfo < 0)
                break;

            int[] ufos = new int[NUfo];
            for (int i = 0; i < NUfo; ++i)
                ufos[i] = Integer.parseInt(br.readLine());

            int NShips = Integer.parseInt(br.readLine());
            int[] ships = new int[NShips];
            for (int i = 0; i < NShips; ++i)
                ships[i] = Integer.parseInt(br.readLine());

            int[] ret = algo.makeMoves(ufos, ships);
            System.out.println(ret.length);
            for (int i = 0; i < ret.length; ++i) {
                System.out.println(ret[i]);
            }
            System.out.flush();
        }
      }
      catch (Exception e) {}
    }
}
