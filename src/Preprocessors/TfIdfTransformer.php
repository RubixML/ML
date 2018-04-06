<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;

class TfIdfTransformer implements Preprocessor
{
    /**
     * The inverse document frequency values for each word in the vocabulary.
     *
     * @var array
     */
    protected $idfs;

    /**
     * Calculate the inverse document frequency values.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        $samples = $data->samples();

        $this->idfs = array_fill_keys(array_keys($samples[0]), 0);

        foreach ($samples as $sample) {
            foreach ($sample as $token => $count) {
                if ($count > 0) {
                    $this->idfs[$token]++;
                }
            }
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
            foreach ($sample as $token => &$feature) {
                $feature *= $this->idf[$token];
            }
        }
    }
}
