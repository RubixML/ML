<?php

namespace Rubix\ML\Benchmarks\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Whitespace;

/**
 * @Groups({"Tokenizers"})
 * @BeforeMethods({"setUp"})
 */
class WhitespaceBench
{
    protected const SAMPLE_TEXT = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec at nisl posuere, luctus sapien vel, maximus ex. Curabitur tincidunt, libero at commodo tempor, magna neque malesuada diam, vel blandit metus velit quis magna. Vestibulum auctor libero quam, eu ullamcorper nulla dapibus a. Mauris id ultricies sapien. Integer consequat mi eget vehicula vulputate. Mauris cursus nisi non semper dictum. Quisque luctus ex in tortor laoreet tincidunt. Vestibulum imperdiet purus sit amet sapien dignissim elementum. Mauris tincidunt eget ex eu laoreet. Etiam efficitur quam at purus sagittis hendrerit. Mauris tempus, sem in pulvinar imperdiet, lectus ipsum molestie ante, id semper nunc est sit amet sem. Nulla at justo eleifend, gravida neque eu, consequat arcu. Vivamus bibendum eleifend metus, id elementum orci aliquet ac. Praesent pellentesque nisi vitae tincidunt eleifend. Pellentesque quis ex et lorem laoreet hendrerit ut ac lorem. Aliquam non sagittis est.';

    /**
     * @var \Rubix\ML\Other\Tokenizers\Whitespace;
     */
    protected $tokenizer;

    public function setUp() : void
    {
        $this->tokenizer = new Whitespace();
    }

    /**
     * @Subject
     * @revs(10)
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function tokenize() : void
    {
        $this->tokenizer->tokenize(self::SAMPLE_TEXT);
    }
}
