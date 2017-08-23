//package net.librec.recommender.cf.ranking;
package src.main.java.net.librec.recommender.cf.ranking;

import com.google.common.collect.BiMap;
import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.*;
import net.librec.recommender.AbstractRecommender;
import net.librec.util.Lists;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.math.structure.VectorEntry;



/**
 * This work is implemented based on librec SLIMRecommender which is based on the following paper
 * Xia Ning and George Karypis, <strong>SLIM: Sparse Linear Methods for Top-N Recommender Systems</strong>, ICDM 2011. <br>
 *
 * This method is called Balanced Neighborhood which is extending SLIMUserRecommender based on the following paper
 * Robin Burke, Nasim Sonboli, Masoud Mansoury and Aldo Ordonez-Gauger, <strong>Balanced Neighborhoods for
 * Fairness-aware Collaborative Recommendation</strong> FATREC 2017.<br>
 * written by Nasim Sonboli
 */
@ModelData({"isRanking", "slim", "coefficientMatrix", "trainMatrix", "similarityMatrix", "knn"})
public class BLNSLIMUserRecommender extends AbstractRecommender {
    /**
     * the number of iterations
     */
    protected int numIterations;

    /**
     * W in original paper, a sparse matrix of aggregation coefficients
     */
    private DenseMatrix coefficientMatrix;

    /**
     * user's nearest neighbors for kNN > 0
     */
    private Set<Integer>[] userNNs;

    /**
     * regularization parameters for the L1 or L2 term
     */
    private float regL1Norm, regL2Norm;

    /**
     *This parameter controls the influence of user balance calculation on the overall optimization.
     */
    private float Lambda3;

    /**
     * This vector is a 1 x M vector, and M is the number of users,
     * this vector is filled with either 1 or -1,
     * If a user belongs to the protected group it is +1, otherwise it is -1
     */
    private double[] groupMembershipVector;

    /**
     * number of nearest neighbors
     */
    protected static int knn;

    /**
     * item similarity matrix
     */
    private SymmMatrix similarityMatrix;

    /**
     * users's nearest neighbors for kNN <=0, i.e., all other items
     */
    private Set<Integer> allUsers;

    /**
     *
     */
    private String membershipFilePath;
    /**
     * initialization
     *
     * @throws LibrecException if error occurs
     */
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        knn = conf.getInt("rec.neighbors.knn.number", 50);
        numIterations = conf.getInt("rec.iterator.maximum");
        regL1Norm = conf.getFloat("rec.slim.regularization.l1", 1.0f);
        regL2Norm = conf.getFloat("rec.slim.regularization.l2", 1.0f);
        Lambda3 = conf.getFloat("rec.slim.regularization.user.balance.controller", 20.0f);
        membershipFilePath = conf.get("data.membership.input.path");

        // set it in the configuration file


        coefficientMatrix = new DenseMatrix(numUsers, numUsers);
        // initial guesses: make smaller guesses (e.g., W.init(0.01)) to speed up training
        coefficientMatrix.init();
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
        System.out.println("Done with the similarity Matrix...");


        for(int userIdx = 0; userIdx < this.numUsers; ++userIdx) {
            this.coefficientMatrix.set(userIdx, userIdx, 0.0d);
        } //iterate through all of the users and initialize
        System.out.println("Done initializing the coefficient matrix...");

        createUserNNs();
        System.out.println("Done creating the nearest neighbor matrix...");

        //needs to be added to the configuration file
        //read a file showing if a user belongs to the protected group or not and create an M x M matrix
        //String content = readFile("./FatSLIMRec/data/membership/membership_movielens.txt");
        String content = readFile(membershipFilePath);

        String[] userSplited = content.split("\r\n");

        groupMembershipVector = new double[numUsers];

        //fill in the membership vector by membership numbers (1, -1)
        for (int userIdx = 0; userIdx < numUsers; userIdx++) {
            //each row contains userIdx and membership, take only the membership number
            int membership = Integer.parseInt(userSplited[userIdx].split("\t")[1]);
            groupMembershipVector[userIdx] = membership;
        }
        System.out.println("Done Reading membership file..");
    }

    /**
     * Reading a file
     */
    public String readFile(String filename)
    {
        String content = null;
        File file = new File(filename);
        FileReader reader = null;
        try {
            reader = new FileReader(file);
            char[] chars = new char[(int) file.length()];
            reader.read(chars);
            content = new String(chars);
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if(reader !=null){reader.close();} } catch (IOException ie) {ie.printStackTrace();}
        }
        return content;
    }

    /**
     * train model
     *
     * @throws LibrecException if error occurs
     */
    @Override
    protected void trainModel() throws LibrecException {
        // number of iteration cycles
        for (int iter = 1; iter <= numIterations; iter++) {

            loss = 0.0d;
            // each cycle iterates through one coordinate direction
            for (int userIdx = 0; userIdx < numUsers; userIdx++) {
                // find k-nearest neighbors of each user
                Set<Integer> nearestNeighborCollection = knn > 0 ? userNNs[userIdx] : allUsers;

                //all the ratings of userIdx for all the items
                double[] itemRatingEntries = new double[numItems];

                Iterator<VectorEntry> itemItr = trainMatrix.colIterator(userIdx);
                while (itemItr.hasNext()) {
                    VectorEntry itemRatingEntry = itemItr.next();
                    itemRatingEntries[itemRatingEntry.index()] = itemRatingEntry.get();
                }

                // for each nearest neighbor nearestNeighborItemIdx, update coefficienMatrix by the coordinate
                // descent update rule
                for (Integer nearestNeighborUserIdx : nearestNeighborCollection) { //user nearest neighbors!
                    if (nearestNeighborUserIdx != userIdx) {
                        double gradSum = 0.0d, rateSum = 0.0d, errors = 0.0d, userBalanceSum =0.0d;

                        //ratings of each user on all the other items
                        Iterator<VectorEntry> nnItemRatingItr = trainMatrix.colIterator(nearestNeighborUserIdx);
                        if (!nnItemRatingItr.hasNext()) {
                            continue;
                        }

                        int nnCount = 0;

                        while (nnItemRatingItr.hasNext()) { // now go through the ratings a user has put on items
                            VectorEntry nnItemVectorEntry = nnItemRatingItr.next();
                            int nnItemIdx = nnItemVectorEntry.index();
                            double nnRating = nnItemVectorEntry.get();
                            double rating = itemRatingEntries[nnItemIdx]; // rating of userIdx on nnItemIdx

                            // Error = Actual rating of user on nnItem - prediction of user on nnItem
                            double error = rating - predict(userIdx, nnItemIdx, nearestNeighborUserIdx);


                            // Calculating Sigma(pk . wik)
                            double userbalance = balancePredictor(userIdx, nnItemIdx, nearestNeighborUserIdx);
                            //ui and uk should be excluded.


                            userBalanceSum += userbalance * userbalance; //user balance squared
                            gradSum += nnRating * error;
                            rateSum += nnRating * nnRating; // sigma r^2

                            errors += error * error;
                            nnCount++;
                        }

                        userBalanceSum /= nnCount; // Doubt: user balance sum (?) why should we divide it by nnCount?
                        gradSum /= nnCount;
                        rateSum /= nnCount;

                        errors /= nnCount;



                        double coefficient = coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
                        double nnMembership = groupMembershipVector[userIdx];
                        // Loss function
                        loss += 0.5 * errors + 0.5 * regL2Norm * coefficient * coefficient + regL1Norm * coefficient +
                                0.5 * Lambda3 * userBalanceSum ;


                        /** Implementing Soft Thresholding => S(beta, Lambda1)+
                         * beta = Sigma(r - Sigma(wr)) + Lambda3 * p * Sigma(wp)
                         * & Sigma(r - Sigma(wr)) = gradSum
                         * & nnMembership = p
                         * & Sigma(wp) = userBalanceSum
                         */
                        double beta = gradSum + (Lambda3 * nnMembership * userBalanceSum) ; //adding user balance to the gradsum
                        double update = 0.0d; //weight

                        if (regL1Norm < Math.abs(beta)) {
                            if (beta > 0) {
                                update = (beta - regL1Norm) / (regL2Norm + rateSum + Lambda3);
                            } else {
                                // One doubt: in this case, wij<0, however, the
                                // paper says wij>=0. How to gaurantee that?
                                update = (beta + regL1Norm) / (regL2Norm + rateSum + Lambda3);
                            }
                        }

                        coefficientMatrix.set(nearestNeighborUserIdx, userIdx, update);//update the coefficient
                    }
                }
            }

            if (isConverged(iter) && earlyStop) {
                break;
            }
        }
    }


    /**
     * predict a specific ranking score for user userIdx on item itemIdx.
     *
     * @param userIdx         user index
     * @param itemIdx         item index
     * @param excludedUserIdx excluded user index
     * @return a prediction without the contribution of excluded item
     */
    protected double predict(int userIdx, int itemIdx, int excludedUserIdx) {

        double predictRating = 0;

        Iterator<VectorEntry> userEntryIterator = trainMatrix.rowIterator(itemIdx);

        while (userEntryIterator.hasNext()) { //iterate through the nearest neighbors of a user and calculate the prediction accordingly

            VectorEntry userEntry = userEntryIterator.next();
            int nearestNeighborUserIdx = userEntry.index(); //nn user index
            double nearestNeighborPredictRating = userEntry.get();

            if (userNNs[userIdx].contains(nearestNeighborUserIdx) && nearestNeighborUserIdx != excludedUserIdx) {

                double coeff = coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
                //Calculate the prediction
                predictRating += nearestNeighborPredictRating * coeff;
                //calculate the user balance
                //take p vector, multiply by the coefficients of neighbors (dot product)
            }
        }
        return predictRating;
    }
    /**
     * calculate the balance for each user according to their membership weight and their coefficient
     *  diag(PW) ^ 2
     *  for all of the nnUsers of a user
     */
    protected double balancePredictor(int userIdx, int itemIdx, int excludedUserIdx) {

        double predictBalance = 0;
        Iterator<VectorEntry> userEntryIterator = trainMatrix.rowIterator(itemIdx);
        while (userEntryIterator.hasNext()) {
            //iterate through the nearest neighbors of a user and calculate the prediction accordingly
            VectorEntry userEntry = userEntryIterator.next();
            int nearestNeighborUserIdx = userEntry.index(); //nn user index

            if (userNNs[userIdx].contains(nearestNeighborUserIdx) && nearestNeighborUserIdx != excludedUserIdx) {
                //take p vector, multiply by the coefficients of neighbors (dot product)
                predictBalance += groupMembershipVector[nearestNeighborUserIdx] * coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
            }
        }
        return predictBalance;
    }


    @Override
    protected boolean isConverged(int iter) {
        double delta_loss = lastLoss - loss;
        lastLoss = loss;

        // print out debug info
        if (verbose) {
            String recName = getClass().getSimpleName().toString();
            String info = recName + " iter " + iter + ": loss = " + loss + ", delta_loss = " + delta_loss;
            LOG.info(info);
        }

        return iter > 1 ? delta_loss < 1e-5 : false;
    }


    /**
     * predict a specific ranking score for user userIdx on item itemIdx.
     *
     * @param userIdx user index
     * @param itemIdx item index
     * @return predictive ranking score for user userIdx on item itemIdx
     * @throws LibrecException if error occurs
     */
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        //create item knn list if does not exist, for local offline model
        if (!(null != userNNs && userNNs.length > 0)) {
            createUserNNs();
        }
        return predict(userIdx, itemIdx, -1);
    }


    /**
     * Create user KNN list.
     */
    public void createUserNNs() {
        userNNs = new HashSet[numUsers];

        // find the nearest neighbors for each user based on user similarity???
        List<Map.Entry<Integer, Double>> tempUserSimList;
        if (knn > 0) {
            for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
                SparseVector similarityVector = similarityMatrix.row(userIdx);
                if (knn < similarityVector.size()) {
                    tempUserSimList = new ArrayList<>(similarityVector.size() + 1);
                    Iterator<VectorEntry> simItr = similarityVector.iterator();
                    while (simItr.hasNext()) {
                        VectorEntry simVectorEntry = simItr.next();
                        tempUserSimList.add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get()));
                    }
                    tempUserSimList = Lists.sortListTopK(tempUserSimList, true, knn);
                    userNNs[userIdx] = new HashSet<>((int) (tempUserSimList.size() / 0.5)); // why 0.5??
                    for (Map.Entry<Integer, Double> tempUserSimEntry : tempUserSimList) {
                        userNNs[userIdx].add(tempUserSimEntry.getKey());
                    }
                } else {
                    userNNs[userIdx] = similarityVector.getIndexSet();
                }
            }
        } else {
            allUsers = new HashSet<>(trainMatrix.columns());
        }
    }
}
