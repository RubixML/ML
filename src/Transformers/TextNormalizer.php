<?php

namespace Rubix\ML\Transformers;

/**
 * Text Normalizer
 *
 * This transformer converts all text to lowercase and *optionally* removes
 * extra whitespace.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TextNormalizer implements Transformer
{
    const SPACES_REGEX = '/\s+/';
    const SPACE = ' ';

    /**
     * Should we trim excess whitespace?
     *
     * @var bool
     */
    protected $trim;

    /**
     * @param  bool  $trim
     * @return void
     */
    public function __construct(bool $trim = false)
    {
        $this->trim = $trim;
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature)) {
                    if ($this->trim) {
                        $feature = preg_replace(self::SPACES_REGEX, self::SPACE, trim($feature)) ?: '';
                    }

                    $feature = strtolower($feature);
                }
            }
        }
    }
}
