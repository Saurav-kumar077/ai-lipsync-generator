"""
Wav2Lip Model Architecture

This module contains the Wav2Lip model architecture for lip-sync generation.
Based on the original Wav2Lip implementation: https://github.com/Rudrabha/Wav2Lip
"""

import torch
from torch import nn
from torch.nn import functional as F


class Conv2d(nn.Module):
    """Convolutional block with batch normalization"""
    
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Wav2Lip(nn.Module):
    """
    Wav2Lip Model for Audio-Driven Talking Face Generation
    
    This model takes audio sequences and face images as input and generates
    lip-synced video frames.
    """
    
    def __init__(self):
        super(Wav2Lip, self).__init__()

        # Face encoder blocks
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),
            
            nn.Sequential(
                Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            
            nn.Sequential(
                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),
        ])

        # Audio encoder
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        # Face decoder blocks
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(
                Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 256, kernel_size=3, stride=1, padding=1, residual=True),
            ),

            nn.Sequential(
                Conv2d(512, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ),

            nn.Sequential(
                Conv2d(256, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ),

            nn.Sequential(
                Conv2d(128, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 32, kernel_size=3, stride=1, padding=1, residual=True),
            ),
        ])

        # Output block
        self.output_block = nn.Sequential(
            Conv2d(64, 32, kernel_size=3, stride=1, padding=1, residual=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        ) 

    def forward(self, audio_sequences, face_sequences):
        """
        Forward pass
        
        Args:
            audio_sequences: Audio mel-spectrogram sequences (B, T, 1, 80, 16)
            face_sequences: Face image sequences (B, C, T, H, W) or (B, C, H, W)
            
        Returns:
            Generated face frames with lip movements
        """
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # Encode audio
        audio_embedding = self.audio_encoder(audio_sequences)

        # Encode face
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        # Decode with audio guidance
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(f"Error concatenating features: {x.size()} and {feats[-1].size()}")
                raise e
            
            feats.pop()
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs
