<?php

namespace Rubix\ML\Helpers;

use Rubix\ML\Exceptions\RuntimeException;

use function is_resource;
use function escapeshellarg;
use function proc_open;
use function proc_close;
use function fwrite;
use function fclose;

/**
 * Graphviz
 *
 * An interface to the popular Graphviz program for generating graph images.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class Graphviz
{
    /**
     * Produces an image from a "dot" formatted string.
     *
     * https://graphviz.org/doc/info/lang.html
     * 
     * See https://graphviz.org/docs/outputs/ for supported formats
     *
     * @param string $dot
     * @param string $path
     * @param string $format
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public static function dotToImage(string $dot, string $path, string $format = 'png') : void
    {
        $command = "dot -T{$format} -o " . escapeshellarg($path);

        $descriptorspec = [
            ['pipe', 'r'],
        ];

        $process = proc_open($command, $descriptorspec, $pipes);

        if (!is_resource($process)) {
            throw new RuntimeException('Graphviz is not installed or in the default path.');
        }

        fwrite($pipes[0], $dot);
        fclose($pipes[0]);

        $ret = proc_close($process);

        if ($ret !== 0) {
            throw new RuntimeException("Failed to create image file '$path' (code $ret).");
        }
    }
}
