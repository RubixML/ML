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
        $command = "dot -T{$format}";

        $descriptorspec = [
            ['pipe', 'r'],
            ['pipe', 'w'],
        ];

        $process = proc_open($command, $descriptorspec, $pipes);

        if (!is_resource($process)) {
            throw new RuntimeException('Graphviz is not installed or in the default path.');
        }

        fwrite($pipes[0], $dot);
        fclose($pipes[0]);

        $data = stream_get_contents($pipes[1]);
        fclose($pipes[1]);

        $ret = proc_close($process);

        if ($ret !== 0) {
            throw new RuntimeException("Graphviz failed during execution (code $ret).");
        }

        $success = file_put_contents($path, $data, LOCK_EX);

        if (!$success) {
            throw new RuntimeException("Failed to write image to file at '$path'.");
        }
    }
}
