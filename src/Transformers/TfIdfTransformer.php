<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class TfIdfTransformer implements Transformer
{
    /**
     * The inverse document frequency values for each feature.
     *
     * @var array
     */
    protected $idfs;

    /**
     * Calculate the inverse document frequency values for each feature column.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }

        $this->idfs = array_fill(0, $dataset->numColumns(), 0);

        foreach ($dataset as $sample) {
            foreach ($sample as $column => $feature) {
                if ($feature > 0) {
                    $this->idfs[$column]++;
                }
            }
        }

        foreach ($this->idfs as &$idf) {
            $idf = log10($dataset->numRows() / ($idf + self::EPSILON));
        }
    }

    /**
     * Multiply the term frequency by the inverse document frequency.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as $i => &$feature) {
                $feature *= $this->idfs[$i];
            }
        }
    }
}
