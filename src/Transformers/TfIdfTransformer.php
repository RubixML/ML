<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;

class TfIdfTransformer implements Transformer
{
    /**
     * The inverse document frequency values for each feature.
     *
     * @var array
     */
    protected $idfs;

    /**
     * Calculate the inverse document frequency values for each feature.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->idfs = array_fill(0, $dataset->columns(), 0);

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $i => $count) {
                if ($count > 0) {
                    $this->idfs[$i]++;
                }
            }
        }

        foreach ($this->idfs as &$idf) {
            $idf = log10($dataset->rows() / ($idf + self::EPSILON));
        }
    }

    /**
     * Transform an array of samples into an array of vectors.
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
