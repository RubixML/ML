<?php

namespace Rubix\Engine\Preprocessors;

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
     * @param  array  $samples
     * @param  array|null  $outcomes
     * @return void
     */
    public function fit(array $samples, ?array $outcomes = null) : void
    {
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
