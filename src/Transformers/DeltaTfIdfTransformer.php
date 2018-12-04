<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Vector;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use InvalidArgumentException;
use RuntimeException;

/**
 * Delta TF-IDF Transformer
 * 
 * A supervised TF-IDF (Term Frequency Inverse Document Frequency) Transformer that
 * differentiates between terms used in the context of two opposing classes such as
 * positive or negative sentiment.
 * 
 * > **Note**: This transformer assumes that its input is made up of word frequency
 * vectors such as those created by the Word Count Vectorizer.
 *
 * References:
 * [1] J. Martineau et al. (2009). Delta TFIDF: An Improved Feature Space
 * for Sentiment Analysis.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DeltaTfIdfTransformer extends TfIdfTransformer
{
    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This transformer requires a'
                . ' labeled training set.');
        }

        list($rows, $columns) = $dataset->shape();

        $ones = Vector::ones($columns)->asArray();

        $this->counts[0] = $this->counts[1] = $ones;
        $this->n = $rows;

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function update(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This transformer requires a'
                . ' labeled training set.');
        }

        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $classes = $dataset->possibleOutcomes();

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with exactly 2 possible label values, '
                . (string) count($classes) . ' found.');
        }

        if (is_null($this->counts)) {
            $this->fit($dataset);
            return;
        }

        $strata = array_values($dataset->stratify());

        foreach ($strata as $i => $stratum) {
            $counts = $this->counts[$i];

            foreach ($stratum as $sample) {
                foreach ($sample as $column => $feature) {
                    if ($feature > 0) {
                        $counts[$column]++;
                    }
                }
            }

            $this->counts[$i] = $counts;
        }

        list($dfPos, $dfNeg) = $this->counts;

        $nPos = count($strata[0]) + 1;
        $nNeg = count($strata[1]) + 1;

        $idfs = [];

        foreach ($dfPos as $i => $value) {
            $idfs[] = log($nPos / $value) - log($nNeg / $dfNeg[$i]) + 1.;
        }

        $this->idfs = Vector::quick($idfs);
    }
}
