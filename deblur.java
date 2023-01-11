import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

public class deblur {

    /**
     * forms a circular  point spread function (PSF) according to input parameter radius R:
     * @param outputImg
     * @param filterSize
     * @param R
     * @return
     */
    private static Mat calcPSF(Mat outputImg, Size filterSize, int R) {

        Mat h = new Mat(filterSize, CvType.CV_32F, new Scalar(0));

        Point point = new Point(filterSize.width / 2, filterSize.height / 2);
        Imgproc.circle(h, point, R, new Scalar(255), -1, 8);

        Scalar summa = Core.sumElems(h);

        Core.divide(h, summa, outputImg);

        return outputImg;

    }

    /**
     * The calcWnrFilter() synthesizes the simplified Wiener filter Hw according to the formula: Hw = H/(abs(H)^2 + 1/NSR)
     * The Wiener filter is a way to restore a blurred image.
     * @param inputhPSF
     * @param nsr
     * @return
     */
    private static Mat calcWnrFilter(Mat inputhPSF, double nsr) {
        Mat outputG = new Mat();
        Mat hPSFshifted = fftShift(inputhPSF);

        List<Mat> planes_tmp = new ArrayList<>();

        planes_tmp.add(hPSFshifted.clone());
        planes_tmp.add(Mat.zeros(hPSFshifted.size(), CvType.CV_32F));

        Mat complexI = new Mat();
        Core.merge(planes_tmp, complexI);
        Core.dft(complexI, complexI);
        Core.split(complexI, planes_tmp);

        // per-element add
        Mat denom = new Mat();
        Core.pow(new Mat(Math.abs(planes_tmp.get(0).nativeObj)), 2, denom);
        Core.add(denom, new Scalar(nsr), denom);
        Core.divide(planes_tmp.get(0), denom, outputG);

        // free all temporary memory
        for (Mat m : planes_tmp) if (m != null) m.release();
        hPSFshifted.release();
        complexI.release();
        denom.release();
        inputhPSF.release();

        return outputG;
    }
    private static Mat getFloat(Mat mat) {
        Mat mat1 = new Mat();
        mat.convertTo(mat1, CvType.CV_32FC1);

        return mat1;
    }

    private static Mat filter2DFreq(Mat inputImg, Mat H) {
        Mat outputImg = new Mat();
        List<Mat> planesI = new ArrayList<>();
        List<Mat> planesH = new ArrayList<>();

        planesI.add(inputImg.clone());
        planesI.add(Mat.zeros(inputImg.size(), CvType.CV_32F));
        Mat complexI = new Mat();
        //merges several arrays to make a single multi-channel array.
        // That is, each element of the output array will be a concatenation of the elements of the input arrays
        Core.merge(planesI, complexI);

        Core.dft(complexI, complexI, Core.DFT_SCALE);// scales the result: divide it by the number of array elements.


        planesH.add(H.clone());
        planesH.add(Mat.zeros(H.size(), CvType.CV_32F));
        Mat complexH = new Mat();

        Core.merge(planesH, complexH);

        Mat complexIH = new Mat();
        Core.mulSpectrums(complexI, complexH, complexIH, 0);

        Core.idft(complexIH, complexIH);
        Core.split(complexIH, planesI);//divide multi-channel matrix to several single-channel
        outputImg = planesI.get(0);

        // free all the temporary memory
        inputImg.release();
        complexI.release();
        complexH.release();
        complexIH.release();
        for (Mat m : planesH) if (m != null) m.release();


        return outputImg;

    }

    /**
     * Perform quadrants swap to rearrange the PSF
     * @param inputImg
     * @return
     */
    private static Mat fftShift(Mat inputImg) {
        Mat outputImg = new Mat();
        outputImg = inputImg.clone();
        int cx = outputImg.cols() / 2;
        int cy = outputImg.rows() / 2;

        Mat q0 = new Mat(outputImg, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(outputImg, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(outputImg, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(outputImg, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        // free all the temporary memory
        inputImg.release();
        q0.release();
        q1.release();
        q2.release();
        q3.release();
        tmp.release();

        return outputImg;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = new Mat();
        String filename = "/Users/yqiaow/eclipse-workspace/demo/deblur/original.jpeg";
        // convert to (0-255),normal range for most image and video formats;
        src = Imgcodecs.imread( filename,Imgcodecs.IMREAD_GRAYSCALE);
        // convert to float 32-bit to match the depth in the future (0-1.0)
        src = getFloat(src);

        Rect roi = new Rect(0, 0, src.cols() & -2, src.rows() & -2);

        Mat Hw = new Mat();
        Mat h = new Mat();
        h = calcPSF(h, roi.size(), 53); //modify R first
        Hw = calcWnrFilter(h, 1.0 / 5200.0); // modify nsr after modify R

        Mat imgOut = new Mat();
        imgOut = filter2DFreq(src.submat(roi), Hw);

        // convert CV_32F into 8bits to save or display by multiplying each pixel by 255.
        imgOut.convertTo(imgOut, CvType.CV_8U);
        Core.normalize(imgOut,imgOut, 0, 255, Core.NORM_MINMAX);
        Imgcodecs.imwrite("result.jpg", imgOut);
    }

}

