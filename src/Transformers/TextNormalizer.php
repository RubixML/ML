<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;

class TextNormalizer implements Transformer
{
    /**
     * Should the text be converted to all lowercase?
     *
     * @var bool
     */
    protected $lowercase;

    /**
     * Should we trim excess whitespace?
     *
     * @var bool
     */
    protected $whitespace;

    /**
     * @param  bool  $lowercase
     * @param  bool  $whitespace
     * @return void
     */
    public function __construct(bool $lowercase = true, bool $whitespace = true)
    {
        $this->lowercase = $lowercase;
        $this->whitespace = $whitespace;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        //
    }

    /**
     * Normalize the dataset.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature)) {
                    if ($this->lowercase) {
                        $feature = strtolower($feature);
                    }

                    if ($this->whitespace) {
                        $feature = preg_replace('/\s+/', ' ', trim($feature));
                    }
                }
            }
        }
    }
}
