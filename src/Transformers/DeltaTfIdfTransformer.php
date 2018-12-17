<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use InvalidArgumentException;
use RuntimeException;

/**
 * Delta TF-IDF Transformer
 * 
 * A supervised TF-IDF (Term Frequency Inverse Document Frequency) Transformer that
 * uses class labels to boost the TF-IDFs of terms by how informative they are. Terms
 * that receive high weight are those whose concentration is primary in one class
 * whereas low weighted terms are more evenly distributed among the classes.
 * 
 * > **Note**: This transformer assumes that its input is made up of word frequency
 * vectors such as those created by the Word Count Vectorizer.
 *
 * References:
 * [1] J. Martineau et al. (2009). Delta TFIDF: An Improved Feature Space
 * for Sentiment Analysis.
 * [2] S. Ghosh et al. (2018). Class Specific TF-IDF Boosting for Short-text
 * Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DeltaTfIdfTransformer implements Elastic
{
    /**
     * The class specific term frequencies of each word i.e. the number of
     * times a word appears in the context of a class label.
     * 
     * @var array[]|null
     */
    protected $tfs;

    /**
     * The document frequencies of each word i.e. the number of times a word
     * appeared in a document given the entire corpus.
     *
     * @var int[]|null
     */
    protected $dfs;

    /**
     * The number of times a word appears throughout the entire corpus.
     * 
     * @var int[]
     */
    protected $totals = [
        //
    ];

    /**
     * The number of documents (samples) that have been fitted so far.
     * 
     * @var int|null
     */
    protected $n;

    /**
     * The inverse document frequency values of each feature column.
     *
     * @var float[]|null
     */
    protected $idfs;

    /**
     * The entropy for each term.
     *
     * @var float[]|null
     */
    protected $entropies;

    /**
     * Return the inverse document frequencies calculated during fitting.
     *
     * @return float[]|null
     */
    public function idfs() : ?array
    {
        return $this->idfs;
    }

    /**
     * Return the entropies of each term that were calculated during fitting.
     *
     * @return float[]|null
     */
    public function entropies() : ?array
    {
        return $this->entropies;
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

        $ones = array_fill(0, $dataset->numColumns(), 1);

        $this->tfs = array_fill_keys($classes, $ones);
        $this->dfs = $this->totals = $ones;
        $this->n = 1;

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

        if (is_null($this->tfs) or is_null($this->dfs)) {
            $this->fit($dataset);
            return;
        }

        foreach ($dataset->stratify() as $class => $stratum) {
            $tfs = $this->tfs[$class];

            foreach ($stratum as $sample) {
                foreach ($sample as $column => $feature) {
                    if ($feature > 0) {
                        $tfs[$column] += $feature;

                        $this->dfs[$column]++;
                        $this->totals[$column] += $feature;
                    }
                }
            }

            $this->tfs[$class] = $tfs;
        }

        $this->n += $dataset->numRows();

        $idfs = [];

        foreach ($this->dfs as $df) {
            $idfs[] = log($this->n / $df) + 1.;
        }

        $entropies = array_fill(0, count($this->totals), 0.);

        foreach ($this->tfs as $tfs) {
            foreach ($tfs as $column => $tf) {
                $delta = $tf / $this->totals[$column];

                $entropies[$column] += -$delta * log($delta);
            }
        }

        $this->idfs = $idfs;
        $this->entropies = $entropies;
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
        if (is_null($this->idfs) or is_null($this->entropies)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$feature) {
                if ($feature > 0) {
                    $feature *= $this->idfs[$column];
                    $feature += $this->entropies[$column];
                }
            }
        }
    }
}
