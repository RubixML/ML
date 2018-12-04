<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
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
class DeltaTfIdfTransformer implements Elastic
{
    /**
     * The times a word / feature appeared in a document.
     *
     * @var array|null
     */
    protected $counts;

    /**
     * The inverse document frequency values for each feature column.
     *
     * @var \Rubix\Tensor\Vector|null
     */
    protected $idfs;

    /**
     * The number of documents (samples) that have been fitted so far
     * from the positive stratum.
     * 
     * @var int
     */
    protected $nPos;

    /**
     * The number of documents (samples) that have been fitted so far
     * from the negative stratum.
     * 
     * @var int
     */
    protected $nNeg;

    /**
     * Return the document counts for each word (feature column).
     *
     * @return array|null
     */
    public function counts() : ?array
    {
        return $this->counts;
    }

    /**
     * Return the inverse document frequencies calculated during fitting.
     *
     * @return \Rubix\Tensor\Vector|null
     */
    public function idfs() : ?Vector
    {
        return $this->idfs;
    }

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

        $classes = $dataset->possibleOutcomes();

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with exactly 2 possible label values, '
                . (string) count($classes) . ' found.');
        }

        $ones = Vector::ones($dataset->numColumns())->asArray();

        $this->counts[0] = $this->counts[1] = $ones;
        $this->nPos = $this->nNeg = 1;

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

        $this->nPos += count($strata[0]);
        $this->nNeg += count($strata[1]);

        list($dfPos, $dfNeg) = $this->counts;

        $idfs = [];

        foreach ($dfPos as $i => $df) {
            $idfs[] = log(($this->nPos / $df) / ($this->nNeg / $dfNeg[$i])) + 1.;
        }

        $this->idfs = Vector::quick($idfs);
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
    {
        if (is_null($this->idfs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = Matrix::quick($samples)
            ->multiply($this->idfs)
            ->asArray();
    }
}
