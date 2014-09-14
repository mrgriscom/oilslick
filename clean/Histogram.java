import java.io.*;

public class Histogram {
    public static void main(String[] args) throws IOException {
        for (String path : args) {
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));
            int[] hist = new int[65536];

            while (true) {
                try {
                    hist[in.readShort() + (1<<15)]++;
                } catch (EOFException e) {
                    break;
                }
            }
            in.close();

            System.out.println(path);
            for (int i = 0; i < (1<<16); i++) {
                int x = i - (1<<15);
                int count = hist[i];
                if (count > 0) {
                    System.out.println(x + ": " + count);
                }
            }
        }
    }
}