%data_set=get_vowels();
%ae=split_data(data_set,1);
%fft_ae=fft_data(ae);
%[peaks,location]=find_peak(fft_ae);
%disp(peaks);
%p=plot(fft_ae);
%[data_table,sorted_data_table]=sort_peaks();
%disp(data_table{1,2:end})
%[data_table,ae,ah,aw,eh,ei,er,ih,iy,oa,oo,uh,uw]=sort_peaks();
%{
number=70;

[training_ae,test_ae]=split_matrix(ae,number);
[training_ah,test_ah]=split_matrix(ah,number);
[training_aw,test_aw]=split_matrix(aw,number);
[training_eh,test_eh]=split_matrix(eh,number);
[training_ei,test_ei]=split_matrix(ei,number);
[training_er,test_er]=split_matrix(er,number);
[training_ih,test_ih]=split_matrix(ih,number);
[training_iy,test_iy]=split_matrix(iy,number);
[training_oa,test_oa]=split_matrix(oa,number);
[training_oo,test_oo]=split_matrix(oo,number);
[training_uh,test_uh]=split_matrix(uh,number);
[training_uw,test_uw]=split_matrix(uw,number);

[mean_ae,cov_ae,cov_diag_ae]=find_mean_cov_vectors(training_ae);
[mean_ah,cov_ah,cov_diag_ah]=find_mean_cov_vectors(training_ah);
[mean_aw,cov_aw,cov_diag_aw]=find_mean_cov_vectors(training_aw);
[mean_eh,cov_eh,cov_diag_eh]=find_mean_cov_vectors(training_eh);
[mean_ei,cov_ei,cov_diag_ei]=find_mean_cov_vectors(training_ei);
[mean_er,cov_er,cov_diag_er]=find_mean_cov_vectors(training_er);
[mean_ih,cov_ih,cov_diag_ih]=find_mean_cov_vectors(training_ih);
[mean_iy,cov_iy,cov_diag_iy]=find_mean_cov_vectors(training_iy);
[mean_oa,cov_oa,cov_diag_oa]=find_mean_cov_vectors(training_oa);
[mean_oo,cov_oo,cov_diag_oo]=find_mean_cov_vectors(training_oo);
[mean_uh,cov_uh,cov_diag_uh]=find_mean_cov_vectors(training_uh);
[mean_uw,cov_uw,cov_diag_uw]=find_mean_cov_vectors(training_uw);

mean_matrix=[mean_ae;mean_ah;mean_aw;mean_eh;mean_ei;mean_er;mean_ih;mean_iy;mean_oa;mean_oo;mean_uh;mean_uw];
cov_matrix=[cov_ae;cov_ah;cov_aw;cov_eh;cov_ei;cov_er;cov_ih;cov_iy;cov_oa;cov_oo;cov_uh;cov_uw];
cov_diag_matrix=[cov_diag_ae;cov_diag_ah;cov_diag_aw;cov_diag_eh;cov_diag_ei;cov_diag_er;cov_diag_ih;cov_diag_iy;cov_diag_oa;cov_diag_oo;cov_diag_uh;cov_diag_uw];

 
result_ae=check_test(test_ae,mean_matrix,cov_matrix);
result_ah=check_test(test_ah,mean_matrix,cov_matrix);
result_aw=check_test(test_aw,mean_matrix,cov_matrix);
result_eh=check_test(test_eh,mean_matrix,cov_matrix);
result_ei=check_test(test_ei,mean_matrix,cov_matrix);
result_er=check_test(test_er,mean_matrix,cov_matrix);
result_ih=check_test(test_ih,mean_matrix,cov_matrix);
result_iy=check_test(test_iy,mean_matrix,cov_matrix);
result_oa=check_test(test_oa,mean_matrix,cov_matrix);
result_oo=check_test(test_oo,mean_matrix,cov_matrix);
result_uh=check_test(test_uh,mean_matrix,cov_matrix);
result_uw=check_test(test_uw,mean_matrix,cov_matrix);
%}
confusion_matrix=[result_ae,result_ah,result_aw,result_eh,result_ei,result_er,result_ih,result_iy,result_oa,result_oo,result_uh,result_uw
    ];


classLabels = {'ae','ah','aw','eh','er','ei','ih','iy','oa','oo','uh','uw'};
confusionchart(confusion_matrix.',classLabels);
title('confusion matrix for full covariance matrix')


figure;

error_rate=1-sum(diag(confusion_matrix))/sum(confusion_matrix(:));
disp('Confusion matrix for full covariance matrix: ')
disp(confusion_matrix);
disp('the error rate is: ')
disp(error_rate)

result_ae_diag=check_test(test_ae,mean_matrix,cov_diag_matrix);
result_ah_diag=check_test(test_ah,mean_matrix,cov_diag_matrix);
result_aw_diag=check_test(test_aw,mean_matrix,cov_diag_matrix);
result_eh_diag=check_test(test_eh,mean_matrix,cov_diag_matrix);
result_ei_diag=check_test(test_ei,mean_matrix,cov_diag_matrix);
result_er_diag=check_test(test_er,mean_matrix,cov_diag_matrix);
result_ih_diag=check_test(test_ih,mean_matrix,cov_diag_matrix);
result_iy_diag=check_test(test_iy,mean_matrix,cov_diag_matrix);
result_oa_diag=check_test(test_oa,mean_matrix,cov_diag_matrix);
result_oo_diag=check_test(test_oo,mean_matrix,cov_diag_matrix);
result_uh_diag=check_test(test_uh,mean_matrix,cov_diag_matrix);
result_uw_diag=check_test(test_uw,mean_matrix,cov_diag_matrix);


confusion_matrix_diag=[result_ae_diag,result_ah_diag,result_aw_diag,result_eh_diag,result_ei_diag,result_er_diag,result_ih_diag,result_iy_diag,result_oa_diag,result_oo_diag,result_uh_diag,result_uw_diag
    ];
confusionchart(confusion_matrix_diag.',classLabels);
title('Confusion matrix for diagonal covariance matrix')


error_rate_diag=1-sum(diag(confusion_matrix_diag))/sum(confusion_matrix_diag(:));
disp('confusion matrix for diagonal covariance matrix: ')
disp(confusion_matrix_diag);
disp('the error rate is: ')
disp(error_rate_diag)
%{
function [peaks,location]=find_peak(vector)
    f=16000*(1:16384/2)/16384;
    [peaks,location]=findpeaks(vector,'MINPEAKDISTANCE',10,'descend');
    peaks=peaks(1:3);
   
    location=location(1:3);
end
function fft_data_set=fft_data(matrix)
    N=16384;
    fft_data_set=abs(fft(matrix(:,1:end),N));
    fft_data_set=fft_data_set(1:length(fft_data_set)/2);
    end
function spesific_vowel_matrix=split_data(data_set,number)
    spesific_vowel_matrix=data_set(:,1*number:139*number);
end
function vowel_matrix=get_vowels()


    folder_path=['C:\Users\Simon\Documents\2024\estimering\klassifisering_prosjekt\Wovels\Wovels\kids'];
    folder = folder_path; 
    wav_files = dir(fullfile(folder, '*.wav'));
    %file_name = fullfile(folder, wav_files(50).name);
    %[audio_data, sample_rate] = audioread(file_name);
    %N=size(audio_data);
    
    %audio_data=[audio_data;zero_vector];
    %disp(size(audio_data));
    ae=[];
    ah=[];
    aw=[];
    eh=[];
    ei=[];
    er=[];
    ih=[];
    iy=[];
    oa=[];
    oo=[];
    uh=[];
    uw=[];
    for i = 1:length(wav_files)
        file_name = fullfile(folder, wav_files(i).name);
        vowel=file_name(end-5:end-4);
        %disp(vowel);
        [audio_data, sample_rate] = audioread(file_name);
        zero_vector=zeros(16384-length(audio_data),1);
        
        switch vowel
            case 'ae'
                ae=[ae,[audio_data;zero_vector]];
                    
            case 'ah'
                ah=[ah,[audio_data;zero_vector]];
            case 'aw'
                aw=[aw,[audio_data;zero_vector]];
            case 'eh'
                eh=[eh,[audio_data;zero_vector]];
            case 'ei'
                ei=[ei,[audio_data;zero_vector]];
            case 'er'
                er=[er,[audio_data;zero_vector]];
            case 'ih'
                ih=[ih,[audio_data;zero_vector]];
            case 'iy'
                iy=[iy,[audio_data;zero_vector]];
            case 'oa'
                oa=[oa,[audio_data;zero_vector]];
            case 'oo'
                oo=[oo,[audio_data;zero_vector]];
            case 'uh'
                uh=[uh,[audio_data;zero_vector]];
            case 'uw'
                uw=[uw,[audio_data;zero_vector]];
            otherwise
                disp('This function does not work')
        end
        
    end
    %vowel_matrix=ae;
    
    vowel_matrix=[ae,ah,aw,eh,ei,er,ih,iy,oa,oo,uh,uw];
end
function peaks=read_data()
    A=importdata('C:\Users\Simon\Documents\2024\estimering\klassifisering_prosjekt\Wovels\Wovels\vowdata_nohead.dat');
    peaks=A.data;
end
%}
function [data_table,ae,ah,aw,eh,ei,er,ih,iy,oa,oo,uh,uw]=sort_peaks()
    % Define the file name
filename = 'C:\Users\Simon\Documents\2024\estimering\klassifisering_prosjekt\Wovels\Wovels\vowdata_nohead – modified.dat';

% Specify the delimiter (space)
delimiter = ' ';
options = detectImportOptions(filename, 'Delimiter', delimiter);

% Read the data into a table
data_table = readtable(filename, options);

  ae=[];
  ah=[];
  aw=[];
  eh=[];
  ei=[];
  er=[];
  ih=[];
  iy=[];
  oa=[];
  oo=[];
  uh=[];
  uw=[];
  for i=1:1668
  switch data_table{i,1}{1}(end-1:end)
            case 'ae'
                ae=[ae,[data_table{i,2:end}].'];
                    
            case 'ah'
                ah=[ah,[data_table{i,2:end}].'];
            case 'aw'
                aw=[aw,[data_table{i,2:end}].'];
            case 'eh'
                eh=[eh,[data_table{i,2:end}].'];
            case 'ei'
                ei=[ei,[data_table{i,2:end}].'];
            case 'er'
                er=[er,[data_table{i,2:end}].'];
            case 'ih'
                ih=[ih,[data_table{i,2:end}].'];
            case 'iy'
                iy=[iy,[data_table{i,2:end}].'];
            case 'oa'
                oa=[oa,[data_table{i,2:end}].'];
            case 'oo'
                oo=[oo,[data_table{i,2:end}].'];
            case 'uh'
                uh=[uh,[data_table{i,2:end}].'];
            case 'uw'
                uw=[uw,[data_table{i,2:end}].'];
            otherwise
                disp('This function does not work')
  end
  end

end

function mean_matrix=find_mean(matrix)


    twenty=[mean(matrix(7,:)),mean(matrix(8,:)),mean(matrix(9,:))];
    fifty=[mean(matrix(10,:)),mean(matrix(11,:)),mean(matrix(12,:))];
    eighty=[mean(matrix(13,:)),mean(matrix(14,:)),mean(matrix(15,:))];
    mean_matrix=[twenty;
        fifty;
        eighty];
end
function cov_matrix=find_covar(matrix)
    %[m_20,m_50,m_80]=find_mean(matrix);
    twenty=cov(matrix(7:9,:).');
    fifty=cov(matrix(10:12,:).');
    eighty=cov(matrix(13:15,:).');
    cov_matrix=[twenty;
        fifty;
        eighty];
end
function group=calculate_group(x)
   value_ae=gaussian_dist(x,mean_ae,cov_ae);
   value_ah=gaussian_dist(x,mean_ah,cov_ah);
   value_ar=gaussian_dist(x,mean_ar,cov_ar);
   value_eh=gaussian_dist(x,mean_eh,cov_eh);
   value_ei=gaussian_dist(x,mean_ei,cov_matrix_ei(1:3,:));
   value_er=gaussian_dist(x,mean_er,cov_matrix_er(1:3,:));
   value_ih=gaussian_dist(x,mean_ih,cov_matrix_ih(1:3,:));
   value_iy=gaussian_dist(x,mean_iy,cov_matrix_iy(1:3,:));
   value_oa=gaussian_dist(x,mean_oa,cov_matrix_oa(1:3,:));
   value_oo=gaussian_dist(x,mean_oo,cov_matrix_oo(1:3,:));
   value_uh=gaussian_dist(x,mean_uh,cov_matrix_uh(1:3,:));
   value_uw=gaussian_dist(x,mean_uw,cov_matrix_uw(1:3,:));
    
    value_vector=[value_ae,value_ah,value_ar,value_eh,value_ei,value_er,value_ih,value_iy,value_oa,value_oo,value_uh,value_uw];
    [max,group]=max(value_vector);


end
function value=gaussian_dist(x,mean,cov)
    value=-log(det(cov))-1/2*(x.'-mean)*inv(cov)*(x.'-mean).';
end
function [mean_vector,cov_vector,cov_diag]=find_mean_cov_vectors(matrix)
    mean_vector=[mean(matrix(1,:)),mean(matrix(2,:)),mean(matrix(3,:))];
    cov_vector=cov(matrix.');
    values=[cov(matrix(1,:)),cov(matrix(2,:)),cov(matrix(3,:))];
    cov_diag=diag(values);
end

function [training,test]=split_matrix(matrix,number)
    training=matrix(3:5,1:number);
    test=matrix(3:5,number+1:end);
end


function result=check_test(matrix,mean_matrix,cov_matrix)
    
    
    for i=1:length(matrix(1,:))
        for j=1:12
        n=3*(j-1)+1;
        value_matrix(j,i)=gaussian_dist(matrix(:,i),mean_matrix(j,:),cov_matrix(n:n+2,:));
        end
    end
    sorted=zeros(12);
    for i=1:69
        [max_value,index]=max(value_matrix(:,i));
        sorted(index)=sorted(index)+1;
        result=sorted(:,1);
    end
end
