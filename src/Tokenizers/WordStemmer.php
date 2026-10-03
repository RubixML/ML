<?php

namespace Rubix\ML\Tokenizers;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Tokenizers\Stemmers\PorterEnglish;
use Rubix\ML\Tokenizers\Stemmers\Stemmer;

/**
 * Word Stemmer
 *
 * Word Stemmer reduces inflected and derived words to their root form using a stemmer. For example, the
 * sentence "Majority voting is likely foolish" might stem to "major vote is like foolish." The
 * Porter English stemmer is used by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WordStemmer extends Word
{
    /**
     * The underlying word stemmer.
     *
     * @var Stemmer
     */
    protected $stemmer;

    /**
     * @param Stemmer|null $stemmer
     */
    public function __construct(?Stemmer $stemmer = null)
    {
        $this->stemmer = $stemmer ?? new PorterEnglish();
    }

    /**
     * Return the underlying word stemmer.
     *
     * @internal
     *
     * @return Stemmer
     */
    public function stemmer() : Stemmer
    {
        return $this->stemmer;
    }

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return string[]
     */
    public function tokenize(string $string) : array
    {
        return array_map([$this->stemmer, 'stem'], parent::tokenize($string));
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Word Stemmer (language: ' . Params::shortName(get_class($this->stemmer)) . ')';
    }
}
