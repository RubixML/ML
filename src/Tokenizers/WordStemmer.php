<?php

namespace Rubix\ML\Tokenizers;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Tokenizers\Stemmers\PorterEnglish;
use Rubix\ML\Tokenizers\Stemmers\Stemmer;

use function preg_match_all;
use function preg_match;
use function count;
use function implode;
use function array_map;

/**
 * Word Stemmer
 *
 * Word Stemmer reduces inflected and derived words to their root form using a stemmer. For example, the
 * sentence "Majority voting is likely foolish" might stem to "major vote is like foolish." The
 * Porter English stemmer is used by default. Tokens containing delimiters such as apostrophes and
 * hyphens have only their word segments stemmed, e.g. "something's" stems to "someth's."
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
        return array_map([$this, 'stemToken'], parent::tokenize($string));
    }

    /**
     * Stem each word segment of a token, preserving any interposed
     * punctuation such as apostrophes and hyphens.
     *
     * @internal
     *
     * @param string $token
     * @return string
     */
    protected function stemToken(string $token) : string
    {
        preg_match_all("/(\w+|[^\w]+)/u", $token, $segments);

        for ($i = 0, $count = count($segments[0]); $i < $count; ++$i) {
            if (preg_match('/^\w+$/u', $segments[0][$i])) {
                $segments[0][$i] = $this->stemmer->stem($segments[0][$i]);
            }
        }

        return implode('', $segments[0]);
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
        return 'Word Stemmer (stemmer: ' . Params::toString($this->stemmer) . ')';
    }
}
